



def load():

	data_types = ['train', 'test','valid']
	dataset_dict = dict()
	for data_type in data_types:

	    # with open('../../datasets/multi_model_twitter_ACL/Twitter_datasets/'+data_type+'.txt') as f:
	    with open('../../datasets/multi_model_twitter_AAAI/'+data_type+'.txt') as f:
	#     with open('hello_train.txt') as f:
	        xy_list = list()
	        tokens = list()
	        tags = list()
	        images = list()
	        for line in f:
	            items = line.split()
	#             print(items,len(items))/
	            if len(items) > 1 and '-DOCSTART-' not in items[0].strip():
	#                 print(items)
	                token, tag = items
	#                 token=cleaning(token)
	#                 if len(token) > 0:
	                tokens.append(token.strip())
	#                 else:
	# # #                 if token[0].isdigit():
	#                    tokens.append('##')
	#                 else:
	#                 tokens.append(token.strip())
	                tags.append(tag.strip())

	            elif len(items) > 0 and items[0].startswith("IMGID"):
	            	images.append(items[0].split(":")[1])

	            elif len(tokens) > 0:
	#                 print("23")
	#                 print(tokens,tags)
	                xy_list.append((tokens, tags,images))
	#                 print(xy_list)
	                tokens = list()
	                tags = list()
	                images = list()
	        dataset_dict[data_type] = xy_list

	for key in dataset_dict:
	    print('Number of samples (sentences) in {:<5}: {}'.format(key, len(dataset_dict[key])))

	print('\nHere is a first two samples from the train part of the dataset:')
	first_two_train_samples = dataset_dict['valid'][:2]
	for n, sample in enumerate(first_two_train_samples):
	    # sample is a tuple of sentence_tokens and sentence_tags
	    tokens, tags,images = sample
	    print('Sentence {}'.format(n))
	    print('Tokens: {}'.format(tokens))
	    print('Tags:   {}'.format(tags))
	    print('Images:   {}'.format(images[0]))



	return dataset_dict



