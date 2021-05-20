export CUDA_VISIBLE_DEVICES=0,1,2,3

python -u bertSentvec.py \
    --task_name=high_level \
    --do_train=true \
    --do_predict=false \
    --do_eval=false \
    --data_dir=/search/odin/guobk/data/bert_semantic/finetuneData_tfrecord/ \
    --vocab_file=/search/odin/guobk/data/model/roberta_zh_l12//vocab.txt \
    --bert_config_file=/search/odin/guobk/data/model/roberta_zh_l12/bert_config.json\
    --init_checkpoint=/search/odin/guobk/data/bert_semantic/model3/model.ckpt-776000 \
    --max_seq_length=128 \
    --train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --output_dir=/search/odin/guobk/data/bert_semantic/model6/ \
    --warmup_proportion=0.1 \
    --eval_batch_size=1 \
    --number_examples=36427136 \
    --num_train_epochs=1 \
    --n_gpus=4 >> log/train.log 2>&1 &