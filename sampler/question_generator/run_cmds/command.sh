
# camera to camera direction
python -m Data_generation.sampler.question_generator.camera_to_camera_direction_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/c2c_dir.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/camera_to_camera_direction \
    --per_room_points 12 --max_items 100000
    
# chain position reasoning
python -m Data_generation.sampler.question_generator.chain_position_reasoning_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/chain_position.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/chain_position_reasoning_mca \
    --per_room_points 12 --max_items 100000

# frame to frame action
python -m Data_generation.sampler.question_generator.frame_frame_action_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/frame_frame_action.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/frame_frame_action_mca \
    --per_room_points 12 --max_items 100000

# middle frame
python -m Data_generation.sampler.question_generator.middle_frame_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/middle_frame.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/middle_frame_mca \
    --per_room_points 12 --max_items 100000

# multi step navigation
python -m Data_generation.sampler.question_generator.multi_step_navigation_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/multi_step_navigation.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/multi_step_navigation_mca \
    --per_room_points 12 --max_items 100000

# nearest object
python -m Data_generation.sampler.question_generator.nearest_object_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/nearest_object.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/nearest_object_mca \
    --per_room_points 12 --max_items 100000

# next frame mca
python -m Data_generation.sampler.question_generator.next_frame_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/next_frame.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/next_frame_mca \
    --per_room_points 12 --max_items 100000

# object after rotation
python -m Data_generation.sampler.question_generator.object_after_rotation_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/object_after_rot.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/object_after_rotation_mca \
    --per_room_points 12 --max_items 100000

# object count mca
python -m Data_generation.sampler.question_generator.object_count_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/object_count.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/object_count_mca \
    --per_room_points 12 --max_items 100000

# object-object distance
python -m Data_generation.sampler.question_generator.object_object_distance_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/object_object_distance.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/object_object_distance_mca \
    --per_room_points 12 --max_items 100000

# object size
python -m Data_generation.sampler.question_generator.object_size_mca \
   --scenes_root /data/jjt/data \
   --out /data/jjt/Result/out_json/object_size.jsonl \
   --out-dir /data/jjt/Result/BenchTasks/object_size_mca \
   --per_room_points 12 --max_items 100000

# relative position mca
python -m Data_generation.sampler.question_generator.relative_position_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/relative_position.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/relative_position_mca \
    --per_room_points 12 --max_items 100000

# view rotation mca
python -m Data_generation.sampler.question_generator.view_rotation_mca \
    --scenes_root /data/jjt/data \
    --out /data/jjt/Result/out_json/view_rot.jsonl \
    --out-dir /data/jjt/Result/BenchTasks/view_rotation_mca \
    --per_room_points 12 --max_items 100000