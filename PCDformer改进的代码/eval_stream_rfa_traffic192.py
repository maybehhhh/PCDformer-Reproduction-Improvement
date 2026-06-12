import os
import argparse
import torch
import numpy as np
import random

from exp.exp_main import Exp_Main


def set_seed(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(2021)

    parser = argparse.ArgumentParser(description='Stream evaluation for RFA Traffic 192')

    parser.add_argument('--is_training', type=int, default=0)
    parser.add_argument('--model_id', type=str, default='RFA_traffic_96_192')
    parser.add_argument('--model', type=str, default='PCDformer')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
    parser.add_argument('--data_path', type=str, default='traffic.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=0)
    parser.add_argument('--pred_len', type=int, default=192)

    parser.add_argument('--bucket_size', type=int, default=4)
    parser.add_argument('--n_hashes', type=int, default=4)
    parser.add_argument('--enc_in', type=int, default=862)
    parser.add_argument('--dec_in', type=int, default=862)
    parser.add_argument('--c_out', type=int, default=862)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--top_k', type=int, default=1)

    parser.add_argument('--attention_type', type=str, default='acr')
    parser.add_argument('--core_num', type=int, default=0)
    parser.add_argument('--route_activation', type=str, default='sparsemax')
    parser.add_argument('--hybrid_direct_threshold', type=int, default=512)

    parser.add_argument('--use_revin', type=int, default=1)
    parser.add_argument('--revin_eps', type=float, default=1e-5)
    parser.add_argument('--use_residual_adapter', type=int, default=1)
    parser.add_argument('--adapter_gate_init', type=float, default=-4.0)

    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=0.00020)
    parser.add_argument('--des', type=str, default='PCDformer_RFA')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--use_amp', action='store_true', default=False)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    setting = (
        f'{args.model_id}_{args.model}_{args.data}_ft{args.features}'
        f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}'
        f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}'
        f'_df{args.d_ff}_fc{args.factor}_attn{args.attention_type}'
        f'_core{args.core_num}_route{args.route_activation}_hd{args.hybrid_direct_threshold}'
        f'_rv{args.use_revin}_ra{args.use_residual_adapter}'
        f'_eb{args.embed}_dt{args.distil}_{args.des}_0'
    )

    print('Setting:')
    print(setting)

    exp = Exp_Main(args)

    ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        print('Checkpoint not found:', ckpt_path)
        print('Try:')
        print("find ./checkpoints -name 'checkpoint.pth' | grep RFA_traffic_96_192")
        raise FileNotFoundError(ckpt_path)

    print('Loading checkpoint:', ckpt_path)
    exp.model.load_state_dict(torch.load(ckpt_path, map_location=exp.device))
    exp.model.eval()

    test_data, test_loader = exp._get_data(flag='test')

    total_sq = 0.0
    total_abs = 0.0
    total_num = 0

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_zeros = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_zeros], dim=1).float().to(exp.device)

            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if args.output_attention:
                outputs = outputs[0]

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            true = batch_y[:, -args.pred_len:, f_dim:]

            diff = outputs - true
            total_sq += torch.sum(diff * diff).item()
            total_abs += torch.sum(torch.abs(diff)).item()
            total_num += diff.numel()

            if (i + 1) % 50 == 0:
                print(f'batch {i+1}/{len(test_loader)} | mse_so_far={total_sq/total_num:.6f}, mae_so_far={total_abs/total_num:.6f}')

    mse = total_sq / total_num
    mae = total_abs / total_num

    print('======================================')
    print(f'STREAM TEST RESULT | mse:{mse}, mae:{mae}')
    print('======================================')


if __name__ == '__main__':
    main()
