# 16-825 Assignment 5

## Q1. Classification Model 

Test accuracy of my best model is `97.23 %`. I have trained this model for `250` iterations. 

### Visualization of Correct Predictions

| Ground Truth Class | Predicted Class |                         Point Cloud                          |
| :----------------: | :-------------: | :----------------------------------------------------------: |
|     **Chair**      |    **Chair**    | ![cls_correct_pred_idx_0_gt_0_pred_0](images/vanilla/cls/correct/cls_correct_pred_idx_0_gt_0_pred_0.gif) |
|      **Vase**      |    **Vase**     | ![cls_correct_pred_idx_616_gt_1_pred_1](images/vanilla/cls/correct/cls_correct_pred_idx_616_gt_1_pred_1.gif) |
|      **Vase**      |    **Vase**     | ![cls_correct_pred_idx_617_gt_1_pred_1](images/vanilla/cls/correct/cls_correct_pred_idx_617_gt_1_pred_1.gif) |
|      **Lamp**      |    **Lamp**     | ![cls_correct_pred_idx_723_gt_2_pred_2](images/vanilla/cls/correct/cls_correct_pred_idx_723_gt_2_pred_2.gif) |
|      **Lamp**      |    **Lamp**     | ![cls_correct_pred_idx_724_gt_2_pred_2](images/vanilla/cls/correct/cls_correct_pred_idx_724_gt_2_pred_2.gif) |

### Visualization of Wrong Predictions

| Ground Truth Class | Predicted Class |                         Point Cloud                          |
| :----------------: | :-------------: | :----------------------------------------------------------: |
|     **Chair**      |    **Lamp**     | ![cls_wrong_pred_idx_0_gt_0_pred_2](images/vanilla/cls/wrong/cls_wrong_pred_idx_0_gt_0_pred_2.gif) |
|      **Vase**      |    **Lamp**     | ![cls_wrong_pred_idx_1_gt_1_pred_2](images/vanilla/cls/wrong/cls_wrong_pred_idx_1_gt_1_pred_2.gif) |
|      **Vase**      |    **Lamp**     | ![cls_wrong_pred_idx_2_gt_1_pred_2](images/vanilla/cls/wrong/cls_wrong_pred_idx_2_gt_1_pred_2.gif) |
|      **Lamp**      |    **Vase**     | ![cls_wrong_pred_idx_13_gt_2_pred_1](images/vanilla/cls/wrong/cls_wrong_pred_idx_13_gt_2_pred_1.gif) |
|      **Lamp**      |    **Vase**     | ![cls_wrong_pred_idx_14_gt_2_pred_1](images/vanilla/cls/wrong/cls_wrong_pred_idx_14_gt_2_pred_1.gif) |

As we can see, the wrong predictions look misleading and ambiguous to even to humans. Also, we need to note here that because of class imbalance we hardly mis-classify chairs. There is only a single wrong prediction for chair, but multiple wrong predictions for vase and lamp.    

## Q2. Segmentation Model

Test accuracy of my best model is `92.08 %`. I have trained this model for `250` iterations. 

| Ground Truth Class | Accuracy |                        GT Point Cloud                        |                    Predicted Point Cloud                     |
| :----------------: | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|     **Chair**      | `95.31%` |     ![gt_exp_idx_0](images/vanilla/seg/gt_exp_idx_0.gif)     |   ![pred_exp_idx_0](images/vanilla/seg/pred_exp_idx_0.gif)   |
|     **Chair**      | `98.51%` | ![gt_exp_idx_1](images/vanilla/seg/gt_exp_idx_1.gif) | ![pred_exp_idx_1](images/vanilla/seg/pred_exp_idx_1.gif) |
|     **Chair**      | `97.8%`  | ![gt_exp_idx_6](images/vanilla/seg/gt_exp_idx_6.gif) | ![pred_exp_idx_6](images/vanilla/seg/pred_exp_idx_6.gif) |
|     **Chair**      | `67.08%` |     ![gt_exp_idx_4](images/vanilla/seg/gt_exp_idx_4.gif)     |   ![pred_exp_idx_4](images/vanilla/seg/pred_exp_idx_4.gif)   |
|     **Chair**      | `71.2%`  |     ![gt_exp_idx_9](images/vanilla/seg/gt_exp_idx_9.gif)     |   ![pred_exp_idx_9](images/vanilla/seg/pred_exp_idx_9.gif)   |
|     **Chair**      | `51.6%`  | ![gt_exp_idx_351](images/vanilla/seg/gt_exp_idx_351.gif) | ![pred_exp_idx_351](images/vanilla/seg/pred_exp_idx_351.gif) |
|     **Chair**      | `54.46%` | ![gt_exp_idx_96](images/vanilla/seg/gt_exp_idx_96.gif) | ![pred_exp_idx_96](images/vanilla/seg/pred_exp_idx_96.gif) |



## Q3. Robustness Analysis 

### Rotate the input point clouds by certain degrees

Test accuracy of my best model is `33.15 %` for classification and `30.09 %` for segmentation.

```python
rotation = True
if rotation:
    rotation = torch.tensor([20, 0, 0])
    R = pytorch3d.transforms.euler_angles_to_matrix(rotation, 'XYZ')
    test_dataloader.dataset.data = (R @ test_dataloader.dataset.data.transpose(1, 2)).transpose(1, 2)
```



| Ground Truth Class | Predicted Class |                         Point Cloud                          |
| :----------------: | :-------------: | :----------------------------------------------------------: |
|     **Chair**      |    **Chair**    | ![cls_correct_pred_idx_1_gt_0_pred_0](images/vanilla/rot/cls_correct_pred_idx_1_gt_0_pred_0.gif) |
|     **Chair**      |    **Lamp**     | ![cls_wrong_pred_idx_4_gt_0_pred_1](images/vanilla/rot/cls_wrong_pred_idx_4_gt_0_pred_1.gif) |
|      **Vase**      |    **Vase**     | ![cls_correct_pred_idx_71_gt_1_pred_1](images/vanilla/rot/cls_correct_pred_idx_71_gt_1_pred_1.gif) |
|      **Vase**      |    **Chair**    | ![cls_wrong_pred_idx_546_gt_1_pred_0](images/vanilla/rot/cls_wrong_pred_idx_546_gt_1_pred_0.gif) |
|      **Lamp**      |    **Lamp**     | ![cls_correct_pred_idx_116_gt_2_pred_2](images/vanilla/rot/cls_correct_pred_idx_116_gt_2_pred_2.gif) |
|      **Lamp**      |    **Vase**     | ![cls_wrong_pred_idx_613_gt_2_pred_1](images/vanilla/rot/cls_wrong_pred_idx_613_gt_2_pred_1.gif) |



| Ground Truth Class | Accuracy |                    GT Point Cloud                    |                  Predicted Point Cloud                   |
| :----------------: | :------: | :--------------------------------------------------: | :------------------------------------------------------: |
|     **Chair**      | `33.36%` | ![gt_exp_idx_0](images/vanilla/rot/gt_exp_idx_0.gif) | ![pred_exp_idx_0](images/vanilla/rot/pred_exp_idx_0.gif) |
|     **Chair**      | `33.92%` | ![gt_exp_idx_2](images/vanilla/rot/gt_exp_idx_2.gif) | ![pred_exp_idx_2](images/vanilla/rot/pred_exp_idx_2.gif) |
|     **Chair**      | `16.51%` | ![gt_exp_idx_3](images/vanilla/rot/gt_exp_idx_3.gif) | ![pred_exp_idx_3](images/vanilla/rot/pred_exp_idx_3.gif) |
|     **Chair**      | `22.31%` | ![gt_exp_idx_6](images/vanilla/rot/gt_exp_idx_6.gif) | ![pred_exp_idx_6](images/vanilla/rot/pred_exp_idx_6.gif) |

As we can see rotating the point clouds decreases the accuracy significantly. The network learns the spatial structure of the point clouds but can not deal when the point clouds are rotated.   

### Different number of points points per object 

Modified `--num_points` when evaluating models in `eval_cls.py` and `eval_seg.py` to `2000`. Test accuracy of my best model when  `num_points` is `2000`is `96.12 %` for classification task and `90.09 %` for segmentation task. 

| Ground Truth Class | Predicted Class |                         Point Cloud                          |
| :----------------: | :-------------: | :----------------------------------------------------------: |
|     **Chair**      |    **Chair**    | ![cls_correct_pred_idx_0_gt_0_pred_0](images/vanilla/no_points/cls_correct_pred_idx_0_gt_0_pred_0.gif) |
|     **Chair**      |    **Lamp**     | ![cls_wrong_pred_idx_0_gt_0_pred_2](images/vanilla/no_points/cls_wrong_pred_idx_0_gt_0_pred_2.gif) |
|      **Vase**      |    **Vase**     | ![cls_correct_pred_idx_617_gt_1_pred_1](images/vanilla/no_points/cls_correct_pred_idx_617_gt_1_pred_1.gif) |
|      **Vase**      |    **Lamp**     | ![cls_wrong_pred_idx_1_gt_1_pred_2](images/vanilla/no_points/cls_wrong_pred_idx_1_gt_1_pred_2.gif) |
|      **Lamp**      |    **Lamp**     | ![cls_correct_pred_idx_741_gt_2_pred_2](images/vanilla/no_points/cls_correct_pred_idx_741_gt_2_pred_2.gif) |
|      **Lamp**      |    **Vase**     | ![cls_wrong_pred_idx_15_gt_2_pred_1](images/vanilla/no_points/cls_wrong_pred_idx_15_gt_2_pred_1.gif) |

| Ground Truth Class | Accuracy |                       GT Point Cloud                       |                    Predicted Point Cloud                     |
| :----------------: | :------: | :--------------------------------------------------------: | :----------------------------------------------------------: |
|     **Chair**      | `33.36%` | ![gt_exp_idx_0](images/vanilla/no_points/gt_exp_idx_0.gif) | ![pred_exp_idx_0](images/vanilla/no_points/pred_exp_idx_0.gif) |
|     **Chair**      | `33.92%` | ![gt_exp_idx_2](images/vanilla/no_points/gt_exp_idx_2.gif) | ![pred_exp_idx_2](images/vanilla/no_points/pred_exp_idx_2.gif) |
|     **Chair**      | `16.51%` | ![gt_exp_idx_3](images/vanilla/no_points/gt_exp_idx_3.gif) | ![pred_exp_idx_3](images/vanilla/no_points/pred_exp_idx_3.gif) |
|     **Chair**      | `22.31%` | ![gt_exp_idx_6](images/vanilla/no_points/gt_exp_idx_6.gif) | ![pred_exp_idx_6](images/vanilla/no_points/pred_exp_idx_6.gif) |

As we can see changing the `num_points` to `2000` doesn't effect the accuracy much. But decreasing it further down might decrease the accuracy. 

## Q4. Expressive architectures

#### Comparison of accuracies of all the models. 

|   PointNet   |  PointNet++   |    DGCNN     | Point Transformers |
| :----------: | :-----------: | :----------: | :----------------: |
| **`97.23%`** | **`98.006%`** | **`98.53%`** |    **`98.11%`**    |

|                         Point Clouds                         | Ground Truth | PointNet | PointNet++ |  DGCNN   | PointFormer |
| :----------------------------------------------------------: | :----------: | :------: | :--------: | :------: | :---------: |
| ![cls_wrong_pred_idx_0_gt_0_pred_2](images/vanilla/no_points/cls_wrong_pred_idx_0_gt_0_pred_2.gif) |  **Chair**   | **Lamp** |  **Lamp**  | **Vase** |  **Vase**   |
| ![cls_wrong_pred_idx_17_gt_2_pred_1](images/cls_wrong_pred_idx_17_gt_2_pred_1.gif) |   **Lamp**   | **Vase** |  **Vase**  | **Lamp** |  **Lamp**   |
| ![cls_wrong_pred_idx_9_gt_1_pred_2](images/cls_wrong_pred_idx_9_gt_1_pred_2.gif) |   **Vase**   | **Vase** |  **Vase**  | **Lamp** |  **Vase**   |

For all the below architectures I only used classification model, and did not use any additional data. This could be the reason we did not see significant difference in accuracies, If we rotate the point clouds like in the previous section, we might see a difference in accuracies. 

### PointNet++

> I have changed the num of points to 2000 from 10000 because of GPU VRAM limitation. 

For PointNet++ we implemented `PointNetSetAbstraction` models that use `farthest_point_sample` and `query_ball_point` functions to sample points in `sample_and_group` function and group points for the point clouds.

### DGCNN

> I have changed the num of points to 2000 from 10000 because of GPU VRAM limitation. 
>

For DGCNN we implemented `knn_graph_feature` function which transforms input features and use `k=20` for the knn. We implement the network by using knn_graph_feature layers followed by conv layers and pooling.  

### Point Transformers

> I have changed the num of points to 2000 from 10000 because of GPU VRAM limitation. 

For Point Transformers we implemented `PointTransformerLayer` which follows transformer like approach. 



