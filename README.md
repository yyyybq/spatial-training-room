# 数据准备

## bench

1. 采样方法1：camera_generation逻辑

   `python -m bench_generation.qa_batch_generator --scene {} --out ./tmp/qa_batch_test_ws.jsonl --out-dir ./tryall --max_items 10 --max_items_per_view 1`

   1. object_object_distance_mca(ok)
   2. nearest_object_mca(ok)
   3. object_size_mca(在调)

2. 采样方法2：屋内合法空间平均采样

   采样：`python -m sampler.generate_view `

   对所有场景的所有物体（除去blank list）生成图片和meta.json：

   `python -m sampler.batch_generate_views --scenes_root /data --out ./out_views --per_room_points 12`

   保存了被遮挡比例，可以用来判断数据质量。

   可调参数：

   1. occ_ratio 被遮挡面积 认为超过0.3就算被遮挡
   2. max_height min_height（防止太低或太高）高度范围
   3. per_room_points 房间内尝试几个采样点
   4. min_dist max_dist 距离目标物体的远近

   适合问题：

   1. next_frame_mca
   2. 相对方位
   3. VST里的问题
   4. 扩展到多个房间之间 应该可以做video的，或者可以做房间内部环顾一周/少量移动的video

   

   

## task

下面的任务感觉都不需要设置标准答案，只需要保证问题能够被解答即可，所以数据不需要给“参考位置”，因为遮挡度等等都可以计算，而且答案不唯一。

### task1

可以调用occlusion中的`occluded_area_on_image`得到目标物体被遮挡的比例和在图上的占比，为了效率，这里的计算不是逐像素的（否则会一次算好几分钟），作为损失函数的计算（要求目标物体被遮挡的比例小+目标物体在图上的占比大）

### task2

沿用camera_generation，但是我感觉预先设置方位，生成的成功率不高，如果根据实际位置推理出方位，可能就很难是“正前方”这种描述，而是有角度或不精准的描述，感觉可以训练的时候试试。

### task 3

虽然被注释掉了但感觉也能做？

### task 4

还是用遮挡率即可

<!-- python -m bench_generation.qa_batch_generator \
  --scene /data/liubinglin/jijiatong/ViewSuite/data\ # path to scene folder, including scene like 0013_840910
  --out ./tmp/qa_batch_test_ws.jsonl \
  --out-dir ./try \
  --max_items 10 \
  --max_items_per_view 2 \
  --render

python -m bench_generation.qa_batch_generator \
  --scene /data/liubinglin/jijiatong/ViewSuite/data\
  --out ./tmp/qa_batch_test_ws.jsonl \
  --out-dir ./try \
  --max_items 10 \
  --max_items_per_view 2 \
  --render
  --question_type action_next_frame_mca
 -->
