import os
import argparse
import time

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from models.definitions.perceptual_loss_net import PerceptualLossNet
from models.definitions.transformer_net import TransformerNet
import utils.utils as utils


def train(training_config):
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare data loader
    train_loader = utils.get_training_data_loader(training_config)

    # prepare neural networks
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNet(requires_grad=False).to(device)

    optimizer = Adam(transformer_net.parameters())

    # Calculate style image's Gram matrices (style representation)
    style_img_path = os.path.join(training_config['style_images_path'], training_config['style_img_name'])
    style_img = utils.prepare_img(style_img_path, target_shape=None, device=device,
                                  batch_size=training_config['batch_size'])
    style_img_set_of_feature_maps = perceptual_loss_net(style_img)
    target_style_representation = [utils.gram_matrix(x) for x in style_img_set_of_feature_maps]

    utils.print_header(training_config)

    acc_content_loss, acc_style_loss, acc_tv_loss = [0., 0., 0.]
    ts = time.time()

    for epoch in range(training_config['num_of_epochs']):
        for batch_id, (content_batch, _) in enumerate(train_loader):
            content_batch = content_batch.to(device)
            stylized_batch = transformer_net(content_batch)

            # Calculate content loss
            content_batch_set_of_feature_maps = perceptual_loss_net(content_batch)
            stylized_batch_set_of_feature_maps = perceptual_loss_net(stylized_batch)

            target_content_representation = content_batch_set_of_feature_maps.relu2_2
            current_content_representation = stylized_batch_set_of_feature_maps.relu2_2
            content_loss = training_config['content_weight'] * torch.nn.MSELoss(reduction='mean')(
                target_content_representation, current_content_representation)

            # Calculate style loss
            style_loss = 0.0
            current_style_representation = [utils.gram_matrix(x) for x in stylized_batch_set_of_feature_maps]
            for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
                style_loss += torch.nn.MSELoss(reduction='mean')(gram_gt, gram_hat)
            style_loss /= len(target_style_representation)
            style_loss *= training_config['style_weight']

            # Total variation loss
            tv_loss = training_config['tv_weight'] * utils.total_variation(stylized_batch)

            # Combine losses
            total_loss = content_loss + style_loss + tv_loss
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            acc_content_loss += content_loss.item()
            acc_style_loss += style_loss.item()
            acc_tv_loss += tv_loss.item()

            if training_config['enable_tensorboard']:
                writer.add_scalar('Loss/content-loss', content_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/style-loss', style_loss.item(), len(train_loader) * epoch + batch_id + 1)
                writer.add_scalar('Loss/tv-loss', tv_loss.item(), len(train_loader) * epoch + batch_id + 1)

                if batch_id % training_config['image_log_freq'] == 0:
                    stylized = utils.post_process_image(stylized_batch[0].detach().to('cpu').numpy())
                    stylized = np.moveaxis(stylized, 2, 0)
                    writer.add_image('stylized_img', stylized, len(train_loader) * epoch + batch_id + 1)

            if training_config['console_log_freq'] and batch_id % training_config['console_log_freq'] == 0:
                print(
                    f'time elapsed={(time.time() - ts) / 60:.2f}[min]|epoch={epoch + 1}|batch=[{batch_id + 1}/{len(train_loader)}]')

            if training_config['checkpoint_freq'] and (batch_id + 1) % training_config['checkpoint_freq'] == 0:
                training_state = utils.get_training_metadata(training_config)
                training_state["state_dict"] = transformer_net.state_dict()
                training_state["optimizer_state"] = optimizer.state_dict()
                ckpt_model_name = f"ckpt_style_{training_config['style_img_name'].split('.')[0]}.pth"
                torch.save(training_state, os.path.join(training_config['checkpoints_path'], ckpt_model_name))

    # Save final model
    training_state = utils.get_training_metadata(training_config)
    training_state["state_dict"] = transformer_net.state_dict()
    training_state["optimizer_state"] = optimizer.state_dict()
    model_name = f"style_{training_config['style_img_name'].split('.')[0]}.pth"
    torch.save(training_state, os.path.join(training_config['model_binaries_path'], model_name))


if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'train2014')
    style_images_path = os.path.join(os.path.dirname(__file__), 'data', 'style-images')
    model_binaries_path = os.path.join(os.path.dirname(__file__), 'models', 'binaries')
    checkpoints_root_path = os.path.join(os.path.dirname(__file__), 'models', 'checkpoints')
    image_size = 256
    batch_size = 4

    parser = argparse.ArgumentParser()
    parser.add_argument(f"--style_img_name", type=str, default='edtaonisl.jpg')
    parser.add_argument("--content_weight", type=float, default=1e0)
    parser.add_argument("--style_weight", type=float, default=4e5)
    parser.add_argument("--tv_weight", type=float, default=0)
    parser.add_argument("--num_of_epochs", type=int, default=2)
    parser.add_argument("--enable_tensorboard", action='store_true')
    parser.add_argument("--image_log_freq", type=int, default=100)
    parser.add_argument("--console_log_freq", type=int, default=500)
    parser.add_argument("--checkpoint_freq", type=int, default=2000)
    args = parser.parse_args()

    checkpoints_path = os.path.join(checkpoints_root_path, args.style_img_name.split('.')[0])
    if args.checkpoint_freq is not None:
        os.makedirs(checkpoints_path, exist_ok=True)

    training_config = {arg: getattr(args, arg) for arg in vars(args)}
    training_config.update({
        'dataset_path': dataset_path,
        'style_images_path': style_images_path,
        'model_binaries_path': model_binaries_path,
        'checkpoints_path': checkpoints_path,
        'image_size': image_size,
        'batch_size': batch_size
    })

    train(training_config)
