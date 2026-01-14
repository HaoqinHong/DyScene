## “几何引导视觉”的条件生成式 4D 场景重建系统

### 1. 数据流：双模态输入 (Dual-Stream Input)
现在的核心思想是：“视觉特征提供纹理底色，几何特征提供结构模具”。

在 training/dataset.py 中，数据被处理为两条平行的流：

流 A：视觉流 (Visual Stream - The "Paint")
    来源：Depth Anything V3 (DA3) 的 Encoder 特征。
    操作：
        从 DA3 提取高维特征图 (Channels=1536 for Giant)。
        修正维度顺序为 (N, C, H, W)。
        使用 grid_sample 将 3D 点投影回 2D 特征图进行采样。
    产物：visual_tokens。包含了“这个点看起来像什么材质”的信息（如毛发、眼睛的反光）。

流 B：几何流 (Geometry Stream - The "Mold")
    来源：Concerto (3D Foundation Model)。
    操作：将 DA3 生成的点云坐标喂给 Concerto。
    产物：scene_tokens (Concerto Tokens)。包含了“这个点属于物体的哪个部位”的信息（如这是腿、这是头）。

### 2/ 模型核心：DiT (Diffusion Transformer) 架构
在 training/model.py 中，网络不再是简单的拼接（Concat），而是采用了 Adaptive Layer Norm (AdaLN) 进行调制。
    主干 (Main Backbone)：处理 Visual Tokens。网络在“画画”，它的画布是视觉特征。
    条件 (Conditioning)：Geometry Tokens + Time ($t$)。
        AdaLN 机制：每一层 Transformer Block 中，几何和时间条件会生成缩放因子 ($\gamma$) 和偏移因子 ($\beta$)，去“调制” (Modulate) 视觉特征的 LayerNorm。
        逻辑含义：几何结构强制约束视觉特征的演变。比如：“这里虽然纹理像毛发（Visual），但几何上它是腿（Geometry），所以请按照腿在时间 $t$ 的方式移动。”

### 3. 生成与解码 (Generation & Decoding)
模型通过两个 Head 输出最终的 3D 高斯球参数：
    静态基座 (Static Base)：由 Visual + Geometry 共同决定初始状态（位置、基础大小、颜色）。
        初始化修正：Scale 偏移量设为 -2.0，让初始高斯球较小，避免画面糊成一团。
    动态残差 (Dynamic Deltas)：由 DiT 主干 输出。
        预测随时间 $t$ 变化的 $\Delta Position$, $\Delta Rotation$, $\Delta Scale$, $\Delta Opacity$。

我们需要彻底改造 model.py。 新的模型不再去学习 base_xyz 或 base_scale，而是直接从输入的 Token 中回归出这些属性。

我们将系统拆解为三个严密的物理空间，确保它们一一对应：
1. 初始化空间 (Initialization Space) —— 解决“东西在哪”
输入: 单目视频帧。
核心组件: Depth Anything V3 (DA3)。
动作: 提取图像的深度图。反投影 (Unproject): 利用相机内参 (K)，把 2D 像素射向 3D 空间，生成点云。坐标系修正 (Critical): 强制将点云放置在相机前方 Z = [1.0, 5.0] 的范围内。相机位置: 锁定相机在原点 (0,0,0)，看向 +Z 方向。

2. 动态演化空间 (Evolution Space) —— 解决“怎么动”
核心模型: FreeTimeGS (DiT)。
输入剥离: 网络不应该看到当前时间 render_t。它只能看到静态的点云 $P_{init}$。
固有属性预测: 网络根据 $P_{init}$，预测出这个粒子生来的属性：
    $V$ (Velocity): 它的初速度向量。
    $t_{center}$ (Peak Time): 它生命最旺盛的时间点。
    $t_{scale}$ (Life Duration): 它能活多久。
确定性演化 (Deterministic Evolution):有了上述属性后，位置和不透明度是由公式算出来的，而不是网络“想”出来的：
    $P(t) = P_{init} + V \times (t - t_{center})$ —— 严格线性运动
    $\alpha(t) = \alpha_{base} \times \exp(-\frac{(t - t_{center})^2}{2\sigma^2})$ —— 时间高斯衰减

3. 渲染成像空间 (Rendering Space) —— 解决“长什么样”
核心组件: gsplat。
输入: 当前时刻的高斯球 + 相机参数。
监督: L1 + SSIM: 像素级对齐。LPIPS: 感知级锐化（去模糊）。DINO: 语义级一致性。


```
python ./training/train.py 2>&1 | tee ./logs/train_log_2026_1_12_ftgs_life_linear.txt
python ./training_freetimegs/inference_video.py
```