# Demo of FIFA for Virtual Try-On ⚽

[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[![Open In Colab][colab-badge]](https://colab.research.google.com/github/hasibzunair/vton-demo/blob/main/demo.ipynb)

This is official code for our BMVC 2022 paper:<br>
[Fill in Fabrics: Body-Aware Self-Supervised Inpainting for Image-Based Virtual Try-On](https://arxiv.org/abs/2210.00918)
<br>

Training details available [fifa-tryon](https://github.com/hasibzunair/fifa-tryon).

<p align="center">
    <a href="#"><img src="./media/vis.png"></a> <br/>
    <em>
    Figure 1. Final try-on outputs of our method with other recent try-on methods.
    </em>
</p>

### Interactive app build using [Gradio](https://gradio.app/)
You can use the model as a simple UI made with gradio. See [gradio_app](https://github.com/dktunited/fifa_demo/tree/master/gradio_app) for details on how to run the app. This app currently works on a local machine with a GPU. Can be hosted on a GPU server.

Note: I attempted to do a CPU implementation first. Got running the try-on and pose estimator models on CPU. The issue is when getting the parsing results (i.e multi-class segmentation) using [this](https://github.com/hasibzunair/Self-Correction-Human-Parsing-for-ACGPN.git) for the person image. The pre-trained model uses [In-Place Activated BatchNorm](https://github.com/mapillary/inplace_abn) for memory-optimized training. The implementation of In-Place Activated BatchNorm is currently only for GPUs. 

### Acknowledgements
This inference codebase is modified from https://github.com/levindabhi/ACGPN to run custom models. The human parser and segmentation models are from https://github.com/hasibzunair/Self-Correction-Human-Parsing-for-ACGPN and https://github.com/hasibzunair/U-2-Net.
