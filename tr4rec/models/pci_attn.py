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


class AttnPredictClickedInview(nn.Module):
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

        self.embedding_dim_of_mulihead = 64
        for i in range(10, 2, -1):
            if self.embedding_dim_of_mulihead % i == 0:
                self.attention_heads = i
                break
        print("ATTENTION HEADS:", self.attention_heads)
        
        self.self_attention = nn.MultiheadAttention(
            self.embedding_dim_of_mulihead, self.attention_heads, batch_first=True, dropout=0.1
        )
        self.normalizer_self_attention = torch.nn.BatchNorm1d(self.max_inview)
        
        self.cross_attention = nn.MultiheadAttention(
            self.embedding_dim_of_mulihead, self.attention_heads, batch_first=True, dropout=0.1
        )
        self.normalizer_cross_attention = torch.nn.BatchNorm1d(self.max_inview)
        
        self.inview_dim = 768
        self.inview_projector = nn.Linear(in_features=self.inview_dim, out_features=64)
        self.classifier_input_dim = 128

        self.classify = nn.Sequential(
            nn.Linear(in_features=self.classifier_input_dim, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
        )
        
        
        # freeze tr4rec
        for block in self.history_model.heads[0].body:
            block.requires_grad = False

    def forward(self, data_for_history, user_features, inviews, length):
        
        # encode history using Tr4Rec
        encoded_history = self.history_model.heads[0].body(data_for_history)

        if self.use_metadata:

            encoded_user_data = torch.div(user_features, 100.0)

            encoded_history = torch.cat(
                (encoded_history, encoded_user_data.cuda()), dim=1
            )

        # get embeddings for inview articles
        encoded_inviews = (
            self.history_model.heads[0]
            .body[0]
            .to_merge.categorical_module.embedding_tables.article_id_fixed.forward(
                inviews
            )
        )

        # project inview articles in the same space as history
        encoded_inviews = self.inview_projector(encoded_inviews)

        # Apply attention mask
        mask = (
            torch.arange(self.max_inview).expand(len(length.cpu()), self.max_inview)
            < length.cpu().unsqueeze(1)
        ).to(encoded_inviews.device)
        # Invert mask for attention
        mask = ~mask
        
        ### FIRST BLOCK ###
        # Self Attention
        encoded_inviews_w_attention = self.self_attention(query=encoded_inviews, key=encoded_inviews, value=encoded_inviews, key_padding_mask=mask)[0]
        encoded_inviews_w_attention = self.normalizer_self_attention(encoded_inviews_w_attention+encoded_inviews)
        
        # Cross Attention
        cross_attn_inview = self.cross_attention(query=encoded_inviews_w_attention, key=encoded_history, value=encoded_history)[0]
        cross_attn_inview = self.normalizer_cross_attention(cross_attn_inview + encoded_inviews_w_attention)

        # Summarize history into a vector
        encoded_history = (
            self.history_model.heads[0]
            .prediction_task_dict["next-item"]
            .sequence_summary(encoded_history)
        )

        # Expand the history vector over all inviews
        encoded_history = (
            encoded_history.unsqueeze(1).expand(-1, encoded_inviews.size(1), -1).cuda()
        )
        attn_output = torch.cat(
            (cross_attn_inview, encoded_history), dim=2
        )

        # Predict the probability of click for each inview article
        assemblied = self.classify(attn_output)

        return assemblied


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def load_model(self, path):
        torch.load_state_dict(torch.load(path))
