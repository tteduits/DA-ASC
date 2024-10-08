for model_type in reversed(models):
    train_data = pd.read_csv('/content/SemEval_Laptop_' + model_type + '.csv')
    test_data = pd.read_csv('/content/SemEval_Laptop_test.csv')

    df_test = make_cls_sep(test_data)
    test_dataset = preprocess_data(df_test, MAX_POSITION_EMBEDDING)

    df_train = make_cls_sep(train_data)
    train_dataset = preprocess_data(df_train, MAX_POSITION_EMBEDDING)

    if model_bool == True:

        for alpha in reversed(alpha_values):
            train_dataset = pd.DataFrame()
            for value in train_data['aspect'].unique():
                df_per_aspect = train_data[train_data['aspect'] == value]

                df_train_aspect = make_cls_sep(df_per_aspect)
                train_dataset_aspect = preprocess_data(df_train_aspect, MAX_POSITION_EMBEDDING)

                df_encoded = pd.DataFrame(
                    {'labels': train_dataset_aspect['labels'], 'input_ids': train_dataset_aspect['input_ids'], 'attention_mask': train_dataset_aspect['attention_mask']})
                mixup_df = pd.DataFrame(columns=['labels', 'input_ids', 'attention_mask'])
                for _ in range(len(df_per_aspect) * ratio):
                    lamda = np.random.beta(alpha, alpha)
                    random_rows = df_encoded.sample(n=2, replace=True).reset_index(drop=True)

                    label_mixup = lamda * random_rows['labels'].iloc[0] + (1 - lamda) * random_rows['labels'].iloc[1]
                    input_ids = torch.tensor((lamda * np.array(random_rows['input_ids'].iloc[0]) +
                                              (1 - lamda) * np.array(random_rows['input_ids'].iloc[1])).tolist(),
                                             dtype=torch.long)

                    attention_mask = (input_ids != 0).long()

                    mixup_row = {
                        'labels': label_mixup,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask
                    }

                    mixup_df = mixup_df._append(mixup_row, ignore_index=True)

                df_combined = pd.concat([mixup_df, df_encoded], axis=0, ignore_index=True)
                train_dataset = pd.concat(([df_combined, train_dataset]))

            train_dataset['labels'] = train_dataset['labels'].apply(map_label)
            train_dataset = train_dataset.to_dict(orient='list')
            train_dataset = Dataset.from_dict(train_dataset)


            lr = 3e-6
            weight = 0.01

            model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
            training_args = TrainingArguments(
                per_device_train_batch_size=10,
                num_train_epochs=3,
                logging_dir='./logs_' + str(lr),
                logging_steps=100,
                save_steps=1000,
                evaluation_strategy="epoch",
                output_dir='./results_' + str(lr),
                learning_rate=lr,
                weight_decay=weight,
                load_best_model_at_end=False,
                save_strategy="epoch",
                save_total_limit=1,
            )

            trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            eval_results = trainer.evaluate(eval_dataset=test_dataset)

            df_result.at[alpha, 'accuracy'] = eval_results['eval_accuracy']
            df_result.at[alpha, 'precision_micro'] = eval_results['eval_precision_micro']
            df_result.at[alpha, 'recall_micro'] = eval_results['eval_recall_micro']
            df_result.at[alpha, 'f1_micro'] = eval_results['eval_f1_micro']
            df_result.at[alpha, 'f1_macro'] = eval_results['eval_f1_macro']


            print(df_result)


    # final_model_dir = EXCEL_FOLDER + '/final_model_xlm' + str(model_type)
    # trainer.save_model(final_model_dir)

    #df_result.to_csv(EXCEL_FOLDER + '/results_SemEval_xlm.csv', index=True)
