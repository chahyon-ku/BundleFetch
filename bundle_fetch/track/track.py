from bundle_fetch.utils import nvtx_range
import torch

def check_and_add_keyframe(frame, keyframes):
    keyframes.append(frame)

def track_thread_target(track_stop, track_queue, mask_model, corr_model, pose_model, gui_queue, recon_queue):
    cuda_stream = torch.cuda.Stream()
    keyframes = []
    i_frame = 0
    prev_frame = None

    while not track_stop.is_set():
        frame, event = track_queue.get()
        print('track', i_frame)
        event.synchronize()
        with torch.cuda.stream(cuda_stream):
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    with nvtx_range('mask_model'):
                        mask = mask_model(frame)
                        frame['mask'] = mask
                    with nvtx_range('corr_model'):
                        corrs = corr_model(frame, prev_frame, keyframes)
                        if corrs is not None:
                            for idx, i_keyframe in enumerate(corrs['i_keyframes']):
                                if i_keyframe == 0:
                                    frame['uv_a'] = corrs['uv_a'][idx]
                                    frame['uv_b'] = corrs['uv_b'][idx]
                                    frame['conf'] = corrs['conf'][idx]
                    with nvtx_range('pose_model'):
                        o_T_c_a, o_T_c_b = pose_model(corrs, frame)
                        frame['o_T_c'] = o_T_c_a[0]
                        
                        if o_T_c_b is not None:
                            for i_pair in range(o_T_c_b.shape[0]):
                                if i_pair == 0:
                                    prev_frame['o_T_c'] = o_T_c_b[i_pair]
                                else:
                                    keyframes[corrs['i_keyframes'][i_pair - 1]]['o_T_c'] = o_T_c_b[i_pair]
                        
                    gui_queue.put(frame)
                    if prev_frame is not None:
                        check_and_add_keyframe(prev_frame, keyframes)

        prev_frame = frame
        i_frame += 1
        # if i_frame == 3:
        #     track_stop.set()