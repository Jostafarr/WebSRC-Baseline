python run.py \
	--train_file ../../../../data/websrc1.0_train_.json \
	--predict_file ../../../../data/websrc1.0_dev_.json \
	--root_dir ../../../../data \
	--model_name_or_path ../../../markuplm-large-finetuned-websrc \
	--output_dir ../../../markuplm-large-finetuned-websrc\
	--do_eval \
	--eval_all_checkpoints \