export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATA_DIR="./images/data/crow_pt"
export OUTPUT_DIR="./embeddings/crow_pt"

accelerate launch diffusers/examples/textual_inversion/textual_inversion.py \
  --num_train_epochs=100 \
  --max_train_steps=3000 \
  --resolution=512 \
  --learning_rate=5.0e-04 \
  --gradient_accumulation_steps=4 \
  --pretrained_model_name_or_path=%MODEL_NAME \
  --train_data_dir=%DATA_DIR \
  --output_dir=%OUT_DIR \
  --num_vectors=1 \
  --placeholder_token="<crow_pt>" \
  --initializer_token=bird \
  --learnable_property=object \
  --no_safe_serialization \