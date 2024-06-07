cd data

mkdir ebnerd_demo
cd ebnerd_demo
wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_demo.zip
unzip ebnerd_demo
rm ebnerd_demo.zip
rm -r __MACOSX/

cd ..

mkdir ebnerd_small
cd ebnerd_small

wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_small.zip
unzip ebnerd_small
rm ebnerd_small.zip
rm -r __MACOSX/

cd ..

mkdir ebnerd_large
cd ebnerd_large

wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_large.zip
unzip ebnerd_large
rm ebnerd_large.zip
rm -r __MACOSX/

cd ..

mkdir ebnerd_testset
cd ebnerd_testset

wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/ebnerd_testset.zip 
unzip ebnerd_testset
rm ebnerd_testset.zip
rm -r __MACOSX/

cd ..

mkdir eb_contrastive_vector
cd eb_contrastive_vector

wget https://ebnerd-dataset.s3.eu-west-1.amazonaws.com/artifacts/Ekstra_Bladet_contrastive_vector.zip
unzip Ekstra_Bladet_contrastive_vector
rm Ekstra_Bladet_contrastive_vector.zip
rm -r __MACOSX/