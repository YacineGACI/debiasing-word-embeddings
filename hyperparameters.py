class Hyperparameters:

    # Filepaths to different word lists
    male_words = "data/wordlist/male_word_file.txt"
    female_words = "data/wordlist/female_word_file.txt"
    neutral_words = "data/wordlist/no_gender_list.tsv"
    stereotype_words = "data/wordlist/stereotype_list.tsv"

    save_model = "saved/"


    word_embedding = 'embeddings/glove.txt'
    emb_binary = False

    dev_num = 20

    embedding_dim = 300
    hidden_dim = 150
    latent_dim = 300

    # Pretraining the autoencoder
    pta_optimizer = 'adam'
    pta_learning_rate = 0.0002
    pta_lr_decay = 1
    pta_max_grad_norm = None
    pta_dev_num = 5000
    pta_epochs = 315
    pta_batch_size = 512


    gpu = True