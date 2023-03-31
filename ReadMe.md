# Others

1. Corpus Preprocessing Example.ipynb : Corpus Preprocessing Example 
2. Plot Confusion Matrix Example.ipynb : Confusion Matrix Code example
2. cosine_annealing_with_warmup.ipynb : cosine_annealing_with_warmup Code example. I edited min values when `start point` `smaller than base min lr`. If anyone who find error, Please Tell me.


# Batch Normalizae in Pytorch
* pytorch에서는 layer를 freezing 해도 Batch normalization이 학습되는 문제가 있음
* https://yjs-program.tistory.com/212?category=804886 참조
```python
# 학습에서 Batch Normalization freeze 하는 방법
for module in model.modules():
    if isinstance(module, nn.BatchNorm2d):
        if hasattr(module, 'weight'):
            module.weight.requires_grad_(False)
        if hasattr(module, 'weight'):
            module.bias.requires_grad_(False)
        module.track_running_stats = False
        module.affine = False
```

# pytorch lightning 에서 Multi GPU 사용 시 출력
* pytorch lightning DDP 사용 시 예제
```python
...
    def validation_step(self, batch, batch_idx):
        mode="val"
        outputs = self.step(batch)

        self.log(f'{mode}_loss', outputs['loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{mode}_acc', outputs['acc'], on_step=False, on_epoch=True, sync_dist=True)

        return {'y_pred': outputs['logits'], 'y_true': batch['labels']}
 ...

    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)  # outputs = [local rank 0, local rank 1]
        # local_rank 0 = {'val_loss': torch.tensor([y_pred_step0, y_pred_step1, ....])
        # if self.trainer.is_global_zero == 0: # only 1 process running
        if self.local_rank == 0:  # only 1 process running
            y_true, y_pred = [], []
            for out_dict in outputs:
                y_true.append(torch.cat([x for x in out_dict['y_true']])) # 3차원이므로 해당값을 2차원으로 축소
                y_pred.append(torch.cat([x for x in out_dict['y_pred']])) 
                # (process 1 step size, batch per step, num_label) => [(all batch size per process x step, num_label), ... ]
            y_true = torch.cat(y_true)  # 위 연산을 통해 계산된 process별로 계산된 값들을 합쳐줌 
            y_pred = torch.cat(y_pred)  # [(data length per process x step, num_label), (data length per process x step, num_label), ... => merge data length, num_label) 

            y_true = y_true.detach().cpu().tolist()
            y_pred = y_pred.detach().cpu().tolist()

            report = classification_report(y_true, y_pred, output_dict=True,
                                           target_names=['슬픔', '즐거움/신남', '흐뭇함(귀여움/예쁨)', '화남/분노', '공포/무서움', '놀람'])

            # logging all things
            for key in report.keys():
                mydict = {"valid_"+key + "_" + k: torch.tensor(v, dtype=torch.float32) for k, v in report[key].items()}
                self.log_dict(mydict, prog_bar=False, rank_zero_only=True)
```

* pytorch lightning에서 LightningModule method인 `self.all_gather` 동작 세부설명
```python
# exampe
def validation_step(self, batch, batch_idx):
...
    return {'logits' : torch.tensor([[0,1,1,0], [1,1,0,1]]), 'labels' : torch.tensor([[1,1,3,4],[4,2,1,2]])}
def validation_epoch_end(self, outputs):
    outputs = self.all_gather   # [process1 outputs, process2 outputs, ....]
                                # process1 = {'logits' : [[[0,1,1,0], [1,1,0,1]], [[....], [...]], ...], 'labels' : ...}
                                # dictionary 형식으로 validation step 반환 시, step의 key는 공유하고 나머지 출력이 합쳐진 형태(tensor 차원 증가)로 합쳐진다
                                # 즉, step 부분에서 2차원으로 반환하면, self.all_gather 동작 시 3차원으로 차원이 한단계 증가하게 된다.
                                # all_gather 이 동작한 경우 [p1 계산결과, p2 결과, p3 결과, ...]
                                # 각 p의 결과는 {key : (process step, batch size, num label)} 가 된다
                                

```
