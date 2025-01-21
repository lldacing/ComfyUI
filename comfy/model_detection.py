import comfy.supported_models
import comfy.supported_models_base
import comfy.utils
import math
import logging
import torch

def count_blocks(state_dict_keys, prefix_string):
    # 初始化计数器
    count = 0

    # 开始一个无限循环，旨在找到不再连续的key编号
    while True:
        # 初始化本轮循环的控制变量c为False
        c = False

        # 遍历状态字典的键，寻找与当前计数匹配的键
        for k in state_dict_keys:
            # 检查键是否以特定格式的字符串开头，该格式基于当前计数值
            if k.startswith(prefix_string.format(count)):
                # 如果找到匹配的键，将c设为True，并中断当前循环
                c = True
                break

        # 如果没有找到匹配的键（即c仍为False），则中断无限循环
        if c == False:
            break

        # 如果找到了匹配的键，则计数器增加，准备下一轮循环
        count += 1

    # 返回最终的计数值，这代表了连续编号的key的数量
    return count

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    # 初始化上下文维度为None
    context_dim = None
    # 初始化是否在Transformer中使用线性层为False
    use_linear_in_transformer = False

    # 构造Transformer块的前缀
    transformer_prefix = prefix + "1.transformer_blocks."
    # 过滤并排序Transformer块的键
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    # 如果存在Transformer块
    if len(transformer_keys) > 0:
        # 计算最后一个Transformer块的深度
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        # 获取上下文维度
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        # 判断是否在Transformer中使用线性层
        use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        # 判断时间堆叠注意力机制是否存在
        time_stack = '{}1.time_stack.0.attn1.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn1.to_q.weight'.format(prefix) in state_dict
        # 判断时间堆叠交叉注意力机制是否存在
        time_stack_cross = '{}1.time_stack.0.attn2.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn2.to_q.weight'.format(prefix) in state_dict
        # 返回相关参数
        return last_transformer_depth, context_dim, use_linear_in_transformer, time_stack, time_stack_cross
    # 如果不存在Transformer块，返回None
    return None

def detect_unet_config(state_dict, key_prefix):
    # 将状态字典的键转换为列表以便后续操作
    state_dict_keys = list(state_dict.keys())

    # 检查是否为mmdit模型，如果是，则按mmdit模型的结构提取配置信息
    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys: #mmdit model
        # 初始化UNet配置字典
        unet_config = {}

        # 提取输入通道数、补丁大小等配置信息
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        patch_size = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size

        # 根据最终层的权重计算输出通道数
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (patch_size * patch_size)

        # 计算UNet的深度
        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        unet_config["input_size"] = None

        # 提取y_embedder的配置信息
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        # 提取context_embedder的配置信息
        context_key = '{}context_embedder.weight'.format(key_prefix)
        if context_key in state_dict_keys:
            in_features = state_dict[context_key].shape[1]
            out_features = state_dict[context_key].shape[0]
            unet_config["context_embedder_config"] = {"target": "torch.nn.Linear", "params": {"in_features": in_features, "out_features": out_features}}

        # 提取num_patches和pos_embed_max_size的配置信息
        num_patches_key = '{}pos_embed'.format(key_prefix)
        if num_patches_key in state_dict_keys:
            num_patches = state_dict[num_patches_key].shape[1]
            unet_config["num_patches"] = num_patches
            unet_config["pos_embed_max_size"] = round(math.sqrt(num_patches))

        # 检查是否使用RMS归一化
        rms_qk = '{}joint_blocks.0.context_block.attn.ln_q.weight'.format(key_prefix)
        if rms_qk in state_dict_keys:
            unet_config["qk_norm"] = "rms"

        # 未使用的配置项
        unet_config["pos_embed_scaling_factor"] = None #unused for inference

        # 提取context_processor的层数
        context_processor = '{}context_processor.layers.0.attn.qkv.weight'.format(key_prefix)
        if context_processor in state_dict_keys:
            unet_config["context_processor_layers"] = count_blocks(state_dict_keys, '{}context_processor.layers.'.format(key_prefix) + '{}.')

        # 提取x_block_self_attn_layers的层数
        unet_config["x_block_self_attn_layers"] = []
        for key in state_dict_keys:
            if key.startswith('{}joint_blocks.'.format(key_prefix)) and key.endswith('.x_block.attn2.qkv.weight'):
                layer = key[len('{}joint_blocks.'.format(key_prefix)):-len('.x_block.attn2.qkv.weight')]
                unet_config["x_block_self_attn_layers"].append(int(layer))

        # 返回UNet配置字典
        return unet_config

    # 检查是否为stable cascade模型，如果是，则按stable cascade模型的结构提取配置信息
    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys: #stable cascade
        # 初始化UNet配置字典
        unet_config = {}
        # 构造文本映射器的名称
        text_mapper_name = '{}clip_txt_mapper.weight'.format(key_prefix)
        # 检查文本映射器是否存在于状态字典中
        if text_mapper_name in state_dict_keys:
            # 如果存在，设置UNet配置为稳定级联阶段'C'
            unet_config['stable_cascade_stage'] = 'c'
            # 获取文本映射器的权重
            w = state_dict[text_mapper_name]
            # 根据权重的形状确定是轻量级还是全量级模型
            if w.shape[0] == 1536: #stage c lite
                # 设置轻量级模型的配置
                unet_config['c_cond'] = 1536
                unet_config['c_hidden'] = [1536, 1536]
                unet_config['nhead'] = [24, 24]
                unet_config['blocks'] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048: #stage c full
                # 设置全量级模型的配置
                unet_config['c_cond'] = 2048
        elif '{}clip_mapper.weight'.format(key_prefix) in state_dict_keys:
            # 如果文本映射器不存在，但剪枝映射器存在，则设置为稳定级联阶段'B'
            unet_config['stable_cascade_stage'] = 'b'
            # 获取下采样块的权重
            w = state_dict['{}down_blocks.1.0.channelwise.0.weight'.format(key_prefix)]
            # 根据权重的形状确定是轻量级还是全量级模型
            if w.shape[-1] == 640:
                # 设置全量级模型的配置
                unet_config['c_hidden'] = [320, 640, 1280, 1280]
                unet_config['nhead'] = [-1, -1, 20, 20]
                unet_config['blocks'] = [[2, 6, 28, 6], [6, 28, 6, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [3, 3, 2, 2]]
            elif w.shape[-1] == 576: #stage b lite
                # 设置轻量级模型的配置
                unet_config['c_hidden'] = [320, 576, 1152, 1152]
                unet_config['nhead'] = [-1, 9, 18, 18]
                unet_config['blocks'] = [[2, 4, 14, 4], [4, 14, 4, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [2, 2, 2, 2]]
        # 返回UNet配置字典
        return unet_config

    # 检查是否为stable audio dit模型，如果是，则按stable audio dit模型的结构提取配置信息
    if '{}transformer.rotary_pos_emb.inv_freq'.format(key_prefix) in state_dict_keys: #stable audio dit
        unet_config = {}
        unet_config["audio_model"] = "dit1.0"
        return unet_config

    if '{}double_layers.0.attn.w1q.weight'.format(key_prefix) in state_dict_keys: #aura flow dit
        unet_config = {}
        unet_config["max_seq"] = state_dict['{}positional_encoding'.format(key_prefix)].shape[1]
        unet_config["cond_seq_dim"] = state_dict['{}cond_seq_linear.weight'.format(key_prefix)].shape[1]
        double_layers = count_blocks(state_dict_keys, '{}double_layers.'.format(key_prefix) + '{}.')
        single_layers = count_blocks(state_dict_keys, '{}single_layers.'.format(key_prefix) + '{}.')
        unet_config["n_double_layers"] = double_layers
        unet_config["n_layers"] = double_layers + single_layers
        return unet_config

    # 检查是否为Hunyuan DiT模型，如果是，则按Hunyuan DiT模型的结构提取配置信息
    if '{}mlp_t5.0.weight'.format(key_prefix) in state_dict_keys: #Hunyuan DiT
        unet_config = {}
        unet_config["image_model"] = "hydit"
        unet_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        unet_config["hidden_size"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0]
        if unet_config["hidden_size"] == 1408 and unet_config["depth"] == 40: #DiT-g/2
            unet_config["mlp_ratio"] = 4.3637
        if state_dict['{}extra_embedder.0.weight'.format(key_prefix)].shape[1] == 3968:
            unet_config["size_cond"] = True
            unet_config["use_style_cond"] = True
            unet_config["image_model"] = "hydit1"
        return unet_config

    if '{}txt_in.individual_token_refiner.blocks.0.norm1.weight'.format(key_prefix) in state_dict_keys: #Hunyuan Video
        dit_config = {}
        dit_config["image_model"] = "hunyuan_video"
        dit_config["in_channels"] = 16
        dit_config["patch_size"] = [1, 2, 2]
        dit_config["out_channels"] = 16
        dit_config["vec_in_dim"] = 768
        dit_config["context_in_dim"] = 4096
        dit_config["hidden_size"] = 3072
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 24
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["axes_dim"] = [16, 56, 56]
        dit_config["theta"] = 256
        dit_config["qkv_bias"] = True
        guidance_keys = list(filter(lambda a: a.startswith("{}guidance_in.".format(key_prefix)), state_dict_keys))
        dit_config["guidance_embed"] = len(guidance_keys) > 0
        return dit_config

    # 检查是否为Flux模型，如果是，则按Flux模型的结构提取配置信息
    if '{}double_blocks.0.img_attn.norm.key_norm.scale'.format(key_prefix) in state_dict_keys: #Flux
        dit_config = {}
        dit_config["image_model"] = "flux"
        dit_config["in_channels"] = 16
        patch_size = 2
        dit_config["patch_size"] = patch_size
        in_key = "{}img_in.weight".format(key_prefix)
        if in_key in state_dict_keys:
            dit_config["in_channels"] = state_dict[in_key].shape[1] // (patch_size * patch_size)
        dit_config["out_channels"] = 16
        dit_config["vec_in_dim"] = 768
        dit_config["context_in_dim"] = 4096
        dit_config["hidden_size"] = 3072
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 24
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["axes_dim"] = [16, 56, 56]
        dit_config["theta"] = 10000
        dit_config["qkv_bias"] = True
        dit_config["guidance_embed"] = "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
        return dit_config

    # 检查是否为Genmo mochi preview模型，如果是，则按Genmo mochi preview模型的结构提取配置信息
    if '{}t5_yproj.weight'.format(key_prefix) in state_dict_keys: #Genmo mochi preview
        dit_config = {}
        dit_config["image_model"] = "mochi_preview"
        dit_config["depth"] = 48
        dit_config["patch_size"] = 2
        dit_config["num_heads"] = 24
        dit_config["hidden_size_x"] = 3072
        dit_config["hidden_size_y"] = 1536
        dit_config["mlp_ratio_x"] = 4.0
        dit_config["mlp_ratio_y"] = 4.0
        dit_config["learn_sigma"] = False
        dit_config["in_channels"] = 12
        dit_config["qk_norm"] = True
        dit_config["qkv_bias"] = False
        dit_config["out_bias"] = True
        dit_config["attn_drop"] = 0.0
        dit_config["patch_embed_bias"] = True
        dit_config["posenc_preserve_area"] = True
        dit_config["timestep_mlp_bias"] = True
        dit_config["attend_to_padding"] = False
        dit_config["timestep_scale"] = 1000.0
        dit_config["use_t5"] = True
        dit_config["t5_feat_dim"] = 4096
        dit_config["t5_token_length"] = 256
        dit_config["rope_theta"] = 10000.0
        return dit_config

    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys and '{}pos_embed.proj.bias'.format(key_prefix) in state_dict_keys:
        # PixArt diffusers
        return None

    # 检查是否为Lightricks ltxv模型
    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys: #Lightricks ltxv
        dit_config = {}
        dit_config["image_model"] = "ltxv"
        return dit_config

    if '{}t_block.1.weight'.format(key_prefix) in state_dict_keys: # PixArt
        patch_size = 2
        dit_config = {}
        dit_config["num_heads"] = 16
        dit_config["patch_size"] = patch_size
        dit_config["hidden_size"] = 1152
        dit_config["in_channels"] = 4
        dit_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')

        y_key = "{}y_embedder.y_embedding".format(key_prefix)
        if y_key in state_dict_keys:
            dit_config["model_max_length"] = state_dict[y_key].shape[0]

        pe_key = "{}pos_embed".format(key_prefix)
        if pe_key in state_dict_keys:
            dit_config["input_size"] = int(math.sqrt(state_dict[pe_key].shape[1])) * patch_size
            dit_config["pe_interpolation"] = dit_config["input_size"] // (512//8) # guess

        ar_key = "{}ar_embedder.mlp.0.weight".format(key_prefix)
        if ar_key in state_dict_keys:
            dit_config["image_model"] = "pixart_alpha"
            dit_config["micro_condition"] = True
        else:
            dit_config["image_model"] = "pixart_sigma"
            dit_config["micro_condition"] = False
        return dit_config

    if '{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix) in state_dict_keys:
        dit_config = {}
        dit_config["image_model"] = "cosmos"
        dit_config["max_img_h"] = 240
        dit_config["max_img_w"] = 240
        dit_config["max_frames"] = 128
        concat_padding_mask = True
        dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = state_dict['{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix)].shape[0]
        dit_config["block_config"] = "FA-CA-MLP"
        dit_config["concat_padding_mask"] = concat_padding_mask
        dit_config["pos_emb_cls"] = "rope3d"
        dit_config["pos_emb_learnable"] = False
        dit_config["pos_emb_interpolation"] = "crop"
        dit_config["block_x_format"] = "THWBD"
        dit_config["affline_emb_norm"] = True
        dit_config["use_adaln_lora"] = True
        dit_config["adaln_lora_dim"] = 256

        if dit_config["model_channels"] == 4096:
            # 7B
            dit_config["num_blocks"] = 28
            dit_config["num_heads"] = 32
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 1.0
            dit_config["rope_w_extrapolation_ratio"] = 1.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        else:  # 5120
            # 14B
            dit_config["num_blocks"] = 36
            dit_config["num_heads"] = 40
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 2.0
            dit_config["rope_w_extrapolation_ratio"] = 2.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_h_extrapolation_ratio"] = 2.0
            dit_config["extra_w_extrapolation_ratio"] = 2.0
            dit_config["extra_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        return dit_config

    # 如果特定键不存在于state_dict_keys中，则返回None
    if '{}input_blocks.0.0.weight'.format(key_prefix) not in state_dict_keys:
        return None

    # 初始化UNet模型的配置字典
    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False
    }

    # 获取输入标签嵌入的权重键，并根据其是否存在更新配置
    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    # 获取模型通道数和输入通道数
    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    # 获取输出通道数
    out_key = '{}out.2.weight'.format(key_prefix)
    if out_key in state_dict:
        out_channels = state_dict[out_key].shape[0]
    else:
        out_channels = 4

    # 初始化各种配置列表
    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    # 初始化视频模型标志
    video_model = False
    video_model_cross = False

    # 初始化当前分辨率和计数器
    current_res = 1
    count = 0

    # 初始化上一个残差块和通道倍增数
    last_res_blocks = 0
    last_channel_mult = 0

    # 计算输入块的数量
    input_block_count = count_blocks(state_dict_keys, '{}input_blocks'.format(key_prefix) + '.{}.')

    # 遍历每个输入块
    for count in range(input_block_count):
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        prefix_output = '{}output_blocks.{}.'.format(key_prefix, input_block_count - count - 1)

        # 获取当前前缀下的所有键
        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        # 获取当前输出前缀下的所有键
        block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))

        # 检查是否为新层
        if "{}0.op.weight".format(prefix) in block_keys: #new layer
            # 将最后一个残差块的数量和通道倍增器添加到相应的列表中
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            # 更新当前分辨率，残差块数量和通道倍增器为下一次迭代做准备
            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0

            # 计算并获取transformer深度，prefix_output是前缀输出，state_dict_keys和state_dict用于查找和计算
            out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            # 根据计算结果更新transformer深度输出列表，如果结果为空则添加0
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            # 构建残差块的前缀，用于后续在block_keys中查找对应的键
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            # 检查残差块的前缀是否在block_keys中，以确定是否需要更新last_res_blocks和last_channel_mult
            if res_block_prefix in block_keys:
                # 如果找到了对应的键，说明这是一个有效的残差块，last_res_blocks计数加1
                last_res_blocks += 1
                # 更新last_channel_mult为当前残差块输出层的通道数
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

                # 调用calculate_transformer_depth函数计算变换器深度，根据返回值更新相关变量
                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    # 如果返回值不为空，添加变换器深度到transformer_depth列表，并根据情况更新context_dim和其他标志变量
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                        video_model_cross = out[4]
                else:
                    # 如果返回值为空，说明当前残差块不涉及变换器，添加0到transformer_depth列表
                    transformer_depth.append(0)

            # 重复上述过程，但这次是为了输出块，使用prefix_output来构建残差块前缀
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                # 如果找到了对应的键，说明这是一个有效的输出残差块，执行类似的逻辑来更新transformer_depth_output
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)


    # 添加最后一个残差块和通道倍增数
    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)

    # 根据中间块的存在情况计算变换器深度
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(state_dict_keys, '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    # 更新UNet配置字典
    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    # 如果是视频模型，添加额外的配置
    if video_model:
        unet_config["extra_ff_mix_layer"] = True
        unet_config["use_spatial_context"] = True
        unet_config["merge_strategy"] = "learned_with_images"
        unet_config["merge_factor"] = 0.0
        unet_config["video_kernel_size"] = [3, 1, 1]
        unet_config["use_temporal_resblock"] = True
        unet_config["use_temporal_attention"] = True
        unet_config["disable_temporal_crossattention"] = not video_model_cross
    else:
        unet_config["use_temporal_resblock"] = False
        unet_config["use_temporal_attention"] = False

    return unet_config

def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in comfy.supported_models.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None

def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    # 从给定的状态字典和UNet键前缀中检测UNet配置
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    # 如果未检测到UNet配置，则返回None
    if unet_config is None:
        return None
    # 根据检测到的UNet配置和状态字典生成模型配置
    model_config = model_config_from_unet_config(unet_config, state_dict)
    # 如果生成的模型配置为空，且允许使用基础配置时，使用基础配置
    if model_config is None and use_base_if_no_match:
        model_config = comfy.supported_models_base.BASE(unet_config)

    # 构造scaled_fp8的键名
    scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
    # 如果scaled_fp8的键存在于状态字典中，提取并处理scaled_fp8权重
    if scaled_fp8_key in state_dict:
        scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
        model_config.scaled_fp8 = scaled_fp8_weight.dtype
        # 如果scaled_fp8权重的数据类型为float32，则将其改为float8_e4m3fn
        if model_config.scaled_fp8 == torch.float32:
            model_config.scaled_fp8 = torch.float8_e4m3fn

    # 返回生成或更新的模型配置
    return model_config

def unet_prefix_from_state_dict(state_dict):
    # 初始化候选模型前缀列表
    candidates = ["model.diffusion_model.", #ldm/sgm models
                  "model.model.", #audio models
                  "net.", #cosmos
                  ]
    # 初始化计数器，用于统计每个候选前缀在state_dict中出现的次数
    counts = {k: 0 for k in candidates}

    # 遍历state_dict中的每个键
    for k in state_dict:
        # 遍历每个候选模型前缀
        for c in candidates:
            # 如果当前键以候选前缀开始，则计数器加一
            if k.startswith(c):
                counts[c] += 1
                # 找到匹配的前缀后立即跳出循环，避免重复计数
                break

    # 找到出现次数最多的模型前缀
    top = max(counts, key=counts.get)
    # 如果最大计数超过5，则返回该前缀，否则返回默认前缀"model."
    if counts[top] > 5:
        return top
    else:
        return "model." #aura flow and others


def convert_config(unet_config):
    # 创建一个新的配置字典，用于后续的定制化设置
    new_config = unet_config.copy()
    # 获取残差块的数量，如果未指定，则使用默认值None
    num_res_blocks = new_config.get("num_res_blocks", None)
    # 获取通道倍增序列，用于确定网络宽度的变化
    channel_mult = new_config.get("channel_mult", None)

    # 如果num_res_blocks是一个整数，而不是列表，将其转换为每个通道倍增对应一个残差块数量的列表
    if isinstance(num_res_blocks, int):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    # 如果配置中指定了注意力解析度，对其进行处理
    if "attention_resolutions" in new_config:
        # 获取注意力解析度并从配置中移除，以便后续处理
        attention_resolutions = new_config.pop("attention_resolutions")
        # 获取转换器深度，如果没有指定，则使用默认值None
        transformer_depth = new_config.get("transformer_depth", None)
        # 获取中间转换器深度，如果没有指定，则使用默认值None
        transformer_depth_middle = new_config.get("transformer_depth_middle", None)

        # 如果transformer_depth是一个整数，而不是列表，将其转换为每个通道倍增对应一个转换器深度的列表
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        # 如果没有指定中间转换器深度，使用最后一个转换器深度作为默认值
        if transformer_depth_middle is None:
            transformer_depth_middle =  transformer_depth[-1]
        # 初始化输入和输出的转换器深度列表
        t_in = []
        t_out = []
        # 初始化解析度缩放因子
        s = 1
        # 遍历每个残差块数量
        for i in range(len(num_res_blocks)):
            res = num_res_blocks[i]
            d = 0
            # 如果当前解析度在注意力解析度列表中，设置当前转换器深度
            if s in attention_resolutions:
                d = transformer_depth[i]

            # 更新输入和输出的转换器深度列表
            t_in += [d] * res
            t_out += [d] * (res + 1)
            # 缩放解析度
            s *= 2
        # 设置最终的转换器深度和输出深度
        transformer_depth = t_in
        # 更新配置字典中的转换器深度和输出深度
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    # 更新配置字典中的残差块数量
    new_config["num_res_blocks"] = num_res_blocks
    # 返回定制化的配置字典
    return new_config


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        res_blocks = count_blocks(state_dict, "down_blocks.{}.resnets.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = count_blocks(state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_refiner = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2560, 'dtype': dtype, 'in_channels': 4, 'model_channels': 384,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [0, 0, 4, 4, 4, 4, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 4,
                    'use_linear_in_transformer': True, 'context_dim': 1280, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2],
            'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_uncliph = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2048, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_unclipl = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 1536, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD15 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_mid_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                     'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                     'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 1,
                     'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_small_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                       'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                       'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 0, 0], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 0,
                       'use_linear_in_transformer': True, 'num_head_channels': 64, 'context_dim': 1, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 9, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_ip2p = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 8, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SSD_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 4, 4], 'transformer_depth_output': [0, 0, 0, 1, 1, 2, 10, 4, 4],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    Segmind_Vega = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 1, 1, 2, 2], 'transformer_depth_output': [0, 0, 0, 1, 1, 1, 2, 2, 2],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_700M = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 5], 'transformer_depth_output': [0, 0, 2, 2, 5, 5],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 6], 'transformer_depth_output': [0, 0, 2, 2, 6, 6],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': 6, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD09_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
            'transformer_depth': [1, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1],
            'use_temporal_attention': False, 'use_temporal_resblock': False, 'disable_self_attentions': [True, False, False]}

    SD_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
            'transformer_depth': [0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': False,
            'context_dim': 768, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 1, 1, 1, 1],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD15_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': dtype, 'in_channels': 9, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}


    supported_models = [SDXL, SDXL_refiner, SD21, SD15, SD21_uncliph, SD21_unclipl, SDXL_mid_cnet, SDXL_small_cnet, SDXL_diffusers_inpaint, SSD_1B, Segmind_Vega, KOALA_700M, KOALA_1B, SD09_XS, SD_XS, SDXL_diffusers_ip2p, SD15_diffusers_inpaint]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return convert_config(unet_config)
    return None

def model_config_from_diffusers_unet(state_dict):
    unet_config = unet_config_from_diffusers_unet(state_dict)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None

def convert_diffusers_mmdit(state_dict, output_prefix=""):
    out_sd = {}

    if 'joint_transformer_blocks.0.attn.add_k_proj.weight' in state_dict: #AuraFlow
        num_joint = count_blocks(state_dict, 'joint_transformer_blocks.{}.')
        num_single = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        sd_map = comfy.utils.auraflow_to_diffusers({"n_double_layers": num_joint, "n_layers": num_joint + num_single}, output_prefix=output_prefix)
    elif 'adaln_single.emb.timestep_embedder.linear_1.bias' in state_dict and 'pos_embed.proj.bias' in state_dict: # PixArt
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        sd_map = comfy.utils.pixart_to_diffusers({"depth": num_blocks}, output_prefix=output_prefix)
    elif 'x_embedder.weight' in state_dict: #Flux
        depth = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth_single_blocks = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        hidden_size = state_dict["x_embedder.bias"].shape[0]
        sd_map = comfy.utils.flux_to_diffusers({"depth": depth, "depth_single_blocks": depth_single_blocks, "hidden_size": hidden_size}, output_prefix=output_prefix)
    elif 'transformer_blocks.0.attn.add_q_proj.weight' in state_dict: #SD3
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth = state_dict["pos_embed.proj.weight"].shape[0] // 64
        sd_map = comfy.utils.mmdit_to_diffusers({"depth": depth, "num_blocks": num_blocks}, output_prefix=output_prefix)
    else:
        return None

    for k in sd_map:
        weight = state_dict.get(k, None)
        if weight is not None:
            t = sd_map[k]

            if not isinstance(t, str):
                if len(t) > 2:
                    fun = t[2]
                else:
                    fun = lambda a: a
                offset = t[1]
                if offset is not None:
                    old_weight = out_sd.get(t[0], None)
                    if old_weight is None:
                        old_weight = torch.empty_like(weight)
                    if old_weight.shape[offset[0]] < offset[1] + offset[2]:
                        exp = list(weight.shape)
                        exp[offset[0]] = offset[1] + offset[2]
                        new = torch.empty(exp, device=weight.device, dtype=weight.dtype)
                        new[:old_weight.shape[0]] = old_weight
                        old_weight = new

                    w = old_weight.narrow(offset[0], offset[1], offset[2])
                else:
                    old_weight = weight
                    w = weight
                w[:] = fun(weight)
                t = t[0]
                out_sd[t] = old_weight
            else:
                out_sd[t] = weight
            state_dict.pop(k)

    return out_sd
