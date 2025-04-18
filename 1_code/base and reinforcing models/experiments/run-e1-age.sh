export HINTS_TRAIN_VAL_TEST_PATH="../../2_pipeline/14_create_train_val_test_datasets_age/store"
export HINTS_NUM_EPOCHS=40
export HINTS_STEPS_PER_EPOCH=100
export HINTS_BATCH_SIZE=20
export HINTS_INPUT_SHAPE="384,384,3"
export HINTS_VERBOSE="False"
export HINTS_VAL_STEPS=50
export HINTS_PRED_STEPS=60
export HINTS_EXPERIMENT_PATH="../../2_pipeline/experiments/out"
export HINTS_EXPERIMENT_ID="E01AGE"
export HINTS_SAVE_MODEL="True"
export HINTS_LOAD_MODEL="False"
export HINTS_LEARNING_RATE=2e-5
export HINTS_MOMENTUM=0

seeds=(1970 1972 2008 2019 2024)
for seed in "${seeds[@]}"; do
  echo "Processing seed: $seed"
  export HINTS_IMAGE_PATH=""
  export HINTS_PREDICTION_FILE="stl_5_10_20_30_35_predictions_$seed.csv"
  export HINTS_TRAIN_FILE="D_5_10_20_30_35_train_$seed.csv"
  export HINTS_VALIDATION_FILE="D_5_10_20_30_35_val_$seed.csv"
  export HINTS_TEST_FILE="test_$seed.csv"
  export HINTS_MODEL_WEIGHTS_FILE="stl_5_10_20_30_35_$seed"
  export HINTS_MODEL_HISTORY_FILE="stl_5_10_20_30_35_acc_and_loss_$seed.csv"
  export HINTS_MODEL_PERFORMANCE_FILE="stl_5_10_20_30_35_performance_$seed.csv"
  export HINTS_SEED=$seed
  python3 ../0_baseline.py
done