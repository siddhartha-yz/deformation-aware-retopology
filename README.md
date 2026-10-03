# 肘部弯了会扁成什么样

弯胳膊时，肘部那一圈会变扁。扁多少，跟线是横着绕还是斜着切，关系不大。

弯 90°，切面大约还剩 **71%**。弯 120°，大约还剩 **一半**。心算：`cos(弯的角度 ÷ 2)`。

![环线和斜线一起弯](docs/figures/bend.gif)

这不是自动重拓扑工具，也没有训练好的模型。它画一条胳膊，让你看见关节切面怎么缩。

## 布线差在哪

横环是一圈一圈绕过肘部。斜线是边斜着跨过去。看上去差很多。

![肘部布线特写](docs/figures/02_wires.png)

## 越弯越扁

下面是环线这一条。百分比是把肘部切开，量面积，再除以没弯时的面积。

![弯曲过程](docs/figures/03_bend.png)

## 两种布线弯到同一个数

![环线和斜线对比](docs/figures/04_compare.png)

切开之后，两条线叠在一起。

![切面](docs/figures/05_sections.png)

![曲线](docs/figures/06_curve.png)

| 弯多少 | 切面大概还剩 |
| ---: | ---: |
| 30° | 96% |
| 45° | 92% |
| 90° | 71% |
| 120° | 50% |

这是两根骨头对半混在一起时的结果（线性混合蒙皮）。不是肌肉模拟，也不是布料。

## 自己跑

```bash
git clone https://github.com/siddhartha-yz/deformation-aware-retopology.git
cd deformation-aware-retopology
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.txt
python demo/armbend.py
```

图写到 `docs/figures/`。看图不需要装 PyTorch。

想放进 Blender 转着看，用这三份模型：

- `docs/meshes/ring_0.obj` 没弯，环线
- `docs/meshes/ring_90.obj` 弯 90°，环线
- `docs/meshes/diagonal_90.obj` 弯 90°，斜线

## 不要指望它做的事

- 不会把雕刻变成能绑骨的低模
- 仓库里的 Blender 插件只是按包围盒铺一个圆柱，不是上面这条胳膊
- 早期 README 里的对比分数不能用，说明在 [STATUS.md](STATUS.md)

静止时的整条胳膊：

![静止](docs/figures/01_rest.png)
