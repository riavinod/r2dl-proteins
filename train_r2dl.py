import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model_classifier, seq_rewriter_gumbel
import argparse
import datasets
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import CategoricalAccuracy, Loss
from ignite.handlers import EarlyStopping
from torch.utils.data import DataLoader
import os
import json
import Sparselandtools
import time
from ksvd import ApproximateKSVD
import ksvd_wrapper   
import class_mapping_utils 
    

def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning_rate')
    parser.add_argument('--temp_min', type=float, default=0.01,
                        help='Temp Min')
    parser.add_argument('--epochs_to_anneal', type=float, default=15.0,
                        help='epochs_to_anneal')
    parser.add_argument('--temp_max', type=float, default=2.0,
                        help='Temp Max')
    parser.add_argument('--reg', type=float, default=0.01,
                        help='regularizer')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='Max Epochs')
    parser.add_argument('--log_every_batch', type=int, default=50,
                        help='Log every batch')
    parser.add_argument('--save_ckpt_every', type=int, default=20,
                        help='Save Checkpoint Every')
    parser.add_argument('--target_dataset', type=str, default="QuestionLabels",
                        help='target_dataset')
    parser.add_argument('--source_dataset', type=str, default="Names",
                        help='source_dataset')
    parser.add_argument('--checkpoints_directory', type=str, default="CKPTS",
                        help='Check Points Directory')
    parser.add_argument('--continue_training', type=str, default="False",
                        help='Continue Training')
    parser.add_argument('--filter_width', type=int, default=5,
                        help='Filter Width')
    parser.add_argument('--hidden_units', type=int, default=256,
                        help='hidden_units')
    parser.add_argument('--embedding_size', type=int, default=256,
                        help='embedding_size')
    parser.add_argument('--resume_run', type=int, default=-1,
                        help='Which run to resume')
    parser.add_argument('--random_network', type=str, default="False",
                        help='Random Network')
    parser.add_argument('--classifier_type', type=str, default="charRNN",
                        help='rnn type')
    parser.add_argument('--print_prob', type=str, default="False",
                        help='Probs')
    parser.add_argument('--progressive', type=str, default="True",
                        help='Progressively increase length for back prop')
    parser.add_argument('--ksvd_iter', type=str, default="True",
                        help='do not exceed 20000')  
    parser.add_argument('--epsilon', type=str, default=1e-7,
                        help='hyperparameter e for k-svd')  
    parser.add_argument('--per_class', type=int, default=1,
                        help='map m source labels to n target labels')  
    

    # TODO create config files

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    base_train_dataset = datasets.get_dataset(args.base_dataset, dataset_type = 'train')

    
    train_dataset = datasets.get_dataset(args.target_dataset, dataset_type = 'train')
    val_dataset = datasets.get_dataset(args.target_dataset, dataset_type = 'val')

    if args.classifier_type == "charRNN":
        lstm_model = model_classifier.uniRNN({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })

    if args.classifier_type == "biRNN":
        lstm_model = model_classifier.biRNN({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })

    if args.classifier_type == "CNN":
        lstm_model = model_classifier.CnnTextClassifier({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })
    
    if args.classifier_type == "BERT":
        model = model_classifier.BERT({
            'vocab_size' : len(base_train_dataset.idx_to_char),
            'hidden_size' : args.hidden_units,
            'target_size' : len(base_train_dataset.classes),
            'embedding_size' : args.embedding_size
        })


    ckpt_dir = "{}/{}_classifer_{}".format(args.checkpoints_directory, args.source_dataset, args.classifier_type)
    ckpt_name = "{}/best_model.pth".format(ckpt_dir)
    if args.random_network != "True":
        model.load_state_dict(torch.load(ckpt_name))
    else:
        pass
    model.eval()
    lstm_loss_criterion = nn.CrossEntropyLoss()

    # initialize sourch model instance here
    source_model = seq_rewriter_gumbel.seq_rewriter({
        'vocab_size' : len(train_dataset.idx_to_char),
        'target_size' : len(base_train_dataset.idx_to_char),
        'filter_width' : args.filter_width,
        'target_sequence_length' : base_train_dataset.seq_length
    })

    r2dl = nn.Sequential(source_model, lstm_model)

    model.to(device)
    source_model.to(device)
    r2dl.to(device)

    parameters = filter(lambda p: p.requires_grad, source_model.parameters())
    for parameter in parameters:
        print("PARAMETERS"), parameter.size()

    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)
    evaluator = create_supervised_evaluator(r2dl,
                                        metrics={
                                            'accuracy': CategoricalAccuracy(),
                                        })
    
    checkpoints_dir = "{}/checkpoints".format(args.checkpoints_directory)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    start_epoch = 0
    training_log = {
        'log' : [],
        'best_epoch' : 0,
        'best_accuracy' : 0.0,
        'running_reward' : []
    }
    running_reward = -args.batch_size

    ksvd = ksvd_wrapper.ksvd()
    dictionary = ksvd.dictInitialization(base_train_dataset, train_dataset)
    V_S = ksvd.constructDictionary(base_train_dataset, train_dataset, epsilon)
    

    
    ce_loss = nn.CrossEntropyLoss()

    if args.continue_training == "True":
        if args.resume_run == -1:
            run_index = len(os.listdir(checkpoints_dir)) - 1
        else:
            run_index = args.resume_run
        checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
        if not os.path.exists(checkpoints_dir):
            raise Exception("Coud not find checkpoints_dir")

        with open("{}/training_log.json".format(checkpoints_dir)) as tlog_f:
            training_log = json.load(tlog_f)

        source_model.load_state_dict(torch.load("{}/best_model.pth".format(checkpoints_dir)))
        start_epoch = training_log['best_epoch']
        # running_reward = training_log['running_reward'][-1]
    else:
        run_index = len(os.listdir(checkpoints_dir))
        checkpoints_dir = "{}/{}".format(checkpoints_dir, run_index)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
    
    temp_min = args.temp_min
    temp_max = args.temp_max

    # define label remapping here
    m_source_classes = len(train_dataset.labels())
    n_prot_labels = 3 # change here for each task
    if args.reduced_labels is not None:
        num_imagenet_labels = args.reduced_labels

    if m_source_classes < n_prot_labels:
        h = class_mapping_utils.create_label_mapping(m_source_classes, args.per_class, n_prot_labels)
    else:
        multi_label_remapper = r2dl.MultiLabelRemapper(n_prot_labels, m_source_classes)
        multi_label_remapper.to(device)
        params = list(r2dl.parameters()) + list(multi_label_remapper.parameters())
    
    optimizer = optim.Adam(params, lr=args.lr)


    for epoch in range(start_epoch, args.max_epochs):
        r2dl.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            source_tokens, labels = batch['input_tokens'], batch['label']
            source_tokens= source_tokens.to(device)
            source_labels = labels.to(device)
            logits = source_model(source_tokens)
            r2dl_logits = class_mapping_utils.get_mapped_logits(logits, h, multi_label_remapper)
            r2dl_loss = ce_loss(r2dl_logits, source_labels)
            r2dl_loss.backward()
            optimizer.step()
            while iter < ksvd_iter:
                ksvd.updateDictionary(train_dataset,dictionary,tokens,V_S)
            batch[0] = ksvd.D*V_S
            for param in r2dl.parameters():
                    param.grad = torch.sign(param.grad)

            if batch_idx % args.log_every_batch == 0:
                if args.print_prob == "True":
                    print ("Temp"), temp, source_model.probs
                print ("Epoch[{}] Iteration[{}] RunningLoss[{}] Reward[{}] Temp[{}]".format(
                    epoch, batch_idx, r2dl_loss, running_reward, temp))

        evaluator.run(train_loader)
        training_metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(epoch, training_metrics['accuracy']))

        evaluator.run(val_loader)
        evaluation_metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f}"
              .format(epoch, evaluation_metrics['accuracy']))

        training_log['log'].append({
            'training_metrics' : training_metrics,
            'evaluation_metrics' : evaluation_metrics,
            'temp' : temp
        })

        if evaluation_metrics['accuracy'] > training_log['best_accuracy']:
            torch.save(source_model.state_dict(), "{}/best_model.pth".format(checkpoints_dir))
            training_log['best_accuracy'] = evaluation_metrics['accuracy']
            training_log['best_epoch'] = epoch

        if epoch % args.save_ckpt_every == 0:
            torch.save(source_model.state_dict(), "{}/model_{}.pth".format(checkpoints_dir, epoch))

        print ("BEST"), training_log['best_epoch'], training_log['best_accuracy']
        with open("{}/training_log.json".format(checkpoints_dir), 'w') as f:
            f.write(json.dumps(training_log))

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)



if __name__ == '__main__':
    main()
