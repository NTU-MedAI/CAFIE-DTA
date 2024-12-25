import argparse


def parse_args():
    parse = argparse.ArgumentParser(description='MODEL INFORMATION')  
    parse.add_argument('--EPOCH', default=500, type=int, help='EOPCH')  
    parse.add_argument('--Learning_rate', default=0.001, type=float, help='Learning rate')
    parse.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parse.add_argument('--accumulation_steps', default=32, type=int, help='accumulation_steps')
    parse.add_argument('--Dropout1', default=0.2, type=float, help='Dropout1')
    parse.add_argument('--Dropout2', default=0.1, type=float, help='Dropout2')
    parse.add_argument('--embedding_size', default=18, type=int, help='embedding_size')
    parse.add_argument('--scheduler_factor', default=0.9, type=float, help='scheduler_factor')
    parse.add_argument('--scheduler_patience', default=5, type=int, help='scheduler_patience')
    parse.add_argument('--dataset', default='kiba', type=str, help='dataset:davis/kiba')
    parse.add_argument('--dataset_path', default='/home/xt/Download/CPA1/fusion/data_kiba/', type=str,
                       help='dataset_path')
    parse.add_argument('--smile_maxlenKB', default=100, type=int, help='smile_maxlenKB')
    parse.add_argument('--proSeq_maxlenKB', default=1000, type=int, help='proSeq_maxlenKB')
    parse.add_argument('--valDV_num', default=300, type=int, help='valDV_num')
    parse.add_argument('--trainKB_num', default=98545, type=int, help='trainKB_num')
    parse.add_argument('--testKB_num', default=19709, type=int, help='testKB_num')
    parse.add_argument('--scheduler_mode', default='min', type=str, help='scheduler_mode')
    parse.add_argument('--scheduler_verbose', default='True', type=str, help='scheduler_verbose')
    parse.add_argument('--logspance_trans', default='False', type=str, help='Davis:True/Kiba:False')

    args = parse.parse_args()  
    return args


args = parse_args()