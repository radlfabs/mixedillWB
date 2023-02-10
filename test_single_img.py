import argparse
import logging
import torch
from src import wb_net
import os.path as path
import os
from src import ops
from src import single_img_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src import weight_refinement as weight_refinement
from tqdm import tqdm



def single_test_net(net, device, input_files, model_name, save_weights,
             multi_scale=False, keep_aspect_ratio=False, t_size=128,
             post_process=False, batch_size=32, wb_settings=None):
  """ Tests a trained network and saves the trained model in harddisk.
  """
  if wb_settings is None:
    wb_settings = ['D', 'S', 'T', 'F', 'C']

  if multi_scale:
    test_set = single_img_dataset.Data(input_files, mode='testing', t_size=t_size,
                            wb_settings=wb_settings,
                            keep_aspect_ratio=keep_aspect_ratio)
  else:
    test_set = single_img_dataset.Data(input_files, mode='testing', t_size=t_size,
                            wb_settings=wb_settings,
                            keep_aspect_ratio=keep_aspect_ratio)

  test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

  
  logging.info(f"""
        Starting testing:
        Model Name:            {model_name}
        Batch size:            {batch_size}
        WB settings:           {wb_settings}
        Save weights:          {save_weights}
        Device:                {device.type}
  """)


  with torch.no_grad():

    for batch in tqdm(test_set, desc="AfifiNet"):

      img = batch['image']

      img = img.to(device=device, dtype=torch.float32)
      _, weights = net(img)
      if multi_scale:
        img_1 = F.interpolate(
          img, size=(int(0.5 * img.shape[2]), int(0.5 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_1 = net(img_1)
        weights_1 = F.interpolate(weights_1, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        img_2 = F.interpolate(
          img, size=(int(0.25 * img.shape[2]), int(0.25 * img.shape[3])),
          mode='bilinear', align_corners=True)
        _, weights_2 = net(img_2)
        weights_2 = F.interpolate(weights_2, size=(img.shape[2], img.shape[3]),
                                 mode='bilinear', align_corners=True)
        weights = (weights + weights_1 + weights_2) / 3

      d_img = batch['fs_d_img']
      d_img = d_img.to(device=device, dtype=torch.float32)
      s_img = batch['fs_s_img']
      s_img = s_img.to(device=device, dtype=torch.float32)
      t_img = batch['fs_t_img']
      t_img = t_img.to(device=device, dtype=torch.float32)
      imgs = [d_img, s_img, t_img]
      f_img = batch['fs_f_img']
      f_img = f_img.to(device=device, dtype=torch.float32)
      imgs.append(f_img)
      c_img = batch['fs_c_img']
      c_img = c_img.to(device=device, dtype=torch.float32)
      imgs.append(c_img)

      weights = F.interpolate(
        weights, size=(d_img.shape[2], d_img.shape[3]),
        mode='bilinear', align_corners=True)

      if post_process:
        for i in range(weights.shape[1]):
          for j in range(weights.shape[0]):
            ref = imgs[0][j, :, :, :]
            curr_weight = weights[j, i, :, :]
            refined_weight = weight_refinement.process_image(ref, curr_weight,
                                                             tensor=True)
            weights[j, i, :, :] = refined_weight
            weights = weights / torch.sum(weights, dim=1)

      for i in range(weights.shape[1]):
        if i == 0:
          out_img = torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
        else:
          out_img += torch.unsqueeze(weights[:, i, :, :], dim=1) * imgs[i]
      result = ops.to_image(out_img.squeeze())
      
      weights_img = []
      if save_weights:
        # save weights
        postfix = ['D', 'S', 'T', 'F', 'C']
        for j in range(weights.shape[1]):
          weight = torch.tile(weights[:, j, :, :], dims=(3, 1, 1))
          weight = ops.to_image(weight)
          weights_img.append(weight)

  logging.info('End of testing')
  return result, weights_img
