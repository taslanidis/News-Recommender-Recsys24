import torch.nn as nn
import torch.nn.functional as F
import torch


class LinearModel(nn.Module):
    def __init__(self, input_feature_dim: int, hidden_dims: int, n_classes: int):
        super(LinearModel, self).__init__()
        layers = []
        layer_sizes = [input_feature_dim] + hidden_dims
        for layer_index in range(1, len(layer_sizes)):
            layers += [
                nn.Dropout(0.1),
                nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]),
                nn.LeakyReLU(),
                nn.BatchNorm1d(layer_sizes[layer_index]),
            ]
        layers += [nn.Linear(layer_sizes[-1], n_classes)]
        self.layers = nn.Sequential(*layers)

        # initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PredictClickedInview(nn.Module):
    def __init__(self, history_model, params):
        necessary_params = [
            "user_feature_dim",
            "max_inview",
            "ovewrite_meta_model",
        ]
        for i in necessary_params:
            if i not in params:
                raise ValueError(
                    "params should be a dictionary with keys " + str(necessary_params)
                )

        self.max_inview = params["max_inview"]
        self.use_summarizer = params["use_summarizer"]
        self.use_metadata = params.get("use_metadata", True)
        
        super().__init__()

        self.history_model = history_model

        # freeze the history model
        for par in self.history_model.parameters():
            par.requires_grad = False

        self.ovewrite_meta_model = params["ovewrite_meta_model"]
        if not params["ovewrite_meta_model"]:
            user_encoded_dim: int = 8
            self.user_meta_data = nn.Linear(
                in_features=params["user_feature_dim"], out_features=user_encoded_dim
            )
        else:
            user_encoded_dim: int = params["user_feature_dim"]

        self.reduce_histor_w_user = nn.Linear(
            in_features=1280 + user_encoded_dim, out_features=768
        )

        self.classifier_input_dim: int = 1280 + user_encoded_dim + 768
        if self.use_summarizer:
            self.classifier_input_dim = 64 + user_encoded_dim + 768

        self.classify = nn.Sequential(
            nn.Linear(in_features=self.classifier_input_dim, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )

        # freeze tr4rec
        for block in self.history_model.heads[0].body:
            block.requires_grad = False

    def forward(self, data_for_history, user_features, inviews, length):
        encoded_history = self.history_model.heads[0].body(data_for_history)

        if self.use_summarizer:
            encoded_history = (
                self.history_model.heads[0]
                .prediction_task_dict["next-item"]
                .sequence_summary(encoded_history)
            )
        else:
            encoded_history = encoded_history.view(user_features.size(0), -1)

        # [B,1] -> [B,64]
        # if not self.ovewrite_meta_model:
        #     encoded_user_data = self.user_meta_data(user_features.float())
        # else:
        #     encoded_user_data = user_features.float()

        if self.use_metadata:

            # normalize age
            encoded_user_data = torch.div(user_features, 100.0)

            # (B,1280) + (B,E_U_D) -> (B,1280+E_U_D) -> (B,1,1280+E_U_D)
            encoded_history = torch.cat(
                (encoded_history, encoded_user_data.cuda()), dim=1
            )

        # reduced_history_user = self.reduce_histor_w_user(
        #     encoded_history_user
        # ).unsqueeze(1)
        # (B, 1, 1280+E_U_D) -> (B, K, 256)
        # reduced_history_user = reduced_history_user.expand(-1, inviews.size(1), -1)

        encoded_history = (
            encoded_history.unsqueeze(1).expand(-1, inviews.size(1), -1).cuda()
        )
        # inviews (B, K, 1) to (B, K, M) embendings
        # get the embendings from the history model
        encoded_inviews = (
            self.history_model.heads[0]
            .body[0]
            .to_merge.categorical_module.embedding_tables.article_id_fixed.forward(
                inviews
            )
        )

        # do pack padded sequence
        # encoded_inviews= torch.nn.utils.rnn.pack_padded_sequence(encoded_inviews, length.cpu(), batch_first=True, enforce_sorted=False)

        # (B, K, 768) + (B, K, 256) -> (B, K, M + 1024)  K is length ov inviews of this batch, M is lentgh of article embedings
        # inviews_history_user = torch.cat((encoded_inviews, reduced_history_user), dim=2)

        # pass through assembly
        # (B, L, 1024) -> (B, L, 62)
        # assemblied = self.assembly(inviews_history_user).squeeze(2)

        # Pack the article embeddings
        # encoded_inviews = torch.nn.utils.rnn.pack_padded_sequence(
        #     encoded_inviews, length.cpu(), batch_first=True, enforce_sorted=False
        # )
        # Apply attention mask
        mask = (
            torch.arange(self.max_inview).expand(len(length.cpu()), self.max_inview)
            < length.cpu().unsqueeze(1)
        ).to(encoded_inviews.device)
        mask = ~mask  # Invert mask for attention

        # For a binary mask, a True value indicates that the corresponding key value will be ignored for the purpose of attention.
        # assemblied = self.cross_attention(query=reduced_history_user, key=encoded_inviews, value=encoded_inviews, key_padding_mask=mask)[0]
        full_concat = torch.cat(
            (encoded_inviews.cuda(), encoded_history.cuda()), dim=2
        )
        assemblied = self.classify(full_concat)
        # (B, L, 62) -> (B, L*62)
        # assemblied = assemblied.view(assemblied.size(0), -1)
        # assemblied = assemblied.squeeze(-1)

        # Unpack the attention output
        # assemblied, _ = torch.nn.utils.rnn.pad_packed_sequence(
        #     torch.nn.utils.pack_padded_sequence(
        #         assemblied, length.cpu(), batch_first=True, enforce_sorted=False
        #     ),
        #     batch_first=True
        # )

        # (B,L*62) -> (B,L)
        # probs_w_padd = self.classify(assemblied)

        return assemblied

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        torch.load_state_dict(torch.load(path))
