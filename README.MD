<img src="assets/banner.gif" width="980"/>

[![arXiv](https://img.shields.io/badge/arXiv-2109.11418v1-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/2109.11418v1)

Based on the [official implementation](https://github.com/ykasten/layered-neural-atlases) from Yoni Kasten of the paper [Layered Neural Atlases for Consistent Video Editing](https://arxiv.org/abs/2109.11418v1), this repository brings some improvements in readability, performance and some new features to this amazing algorithm.

The paper introduces the first approach for neural video unwrapping using an end-to-end optimized interpretable and semantic atlas-based representation, which facilitates easy and intuitive editing in the atlas domain.

## New Features
The new features that this repository brings are:

- **Paper's [Multi Layer Foreground](#multi-layer-foreground-atlases) Atlases feature support. (Thanks to [Yoni Kasten](https://github.com/ykasten/layered-neural-atlases/issues/6))**
- **Support to [Animated Atlases](#generating-atlases).**
- **[Remove Video Foreground or Background Layers](#removing-foreground--background).**
- **Refactored code. (Better readability and understanding of the algorithm)** 
- **Some performance improvements.**
- **More intuitive and easier to use interface.**
- **Option to generate masks from videos with [MODNet: Trimap-Free Portrait Matting in Real Time](#modnet)**
- **Experimental Windows support. (Need to disable ``MaskRCNN`` detectron2 feature and use ``MODNet`` to do mask extraction)**

## Demo Projects
This repository contains 7 demo projects to better understand the algorithm's operation and capabilities. These projects can be downloaded in ZIP format from this [google drive link](https://drive.google.com/file/d/1OwTvo2UuZHqd1yqTvTWgobqTjntrTxkh/view?usp=sharing).

After downloading, just extract the contents of the ZIP file and copy the files to the ``projects`` folder like this:

```
projects
├── bear           
├── blackswan
├── lucia_multi       
├── lucia_single     
├── mallard_water    
├── paragliding   
└── surf
```

## Installation
The installation process is the same as the official repository. The code was tested with python 3.7 and pytorch 1.6. but it may work with other versions.

You can create an anaconda environment called ``neural_atlases`` with the required dependencies by running:
```bash
conda create --name neural_atlases python=3.7 
conda activate neural_atlases 
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 scipy scikit-image tqdm opencv -c pytorch
python -m pip install wandb
```

If you want to use ``MaskRCNN`` for mask extraction (**Linux Only**), you will need to install the ``detectron2`` package:
````bash
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html
````

## Basic Usage
For each new project, new models need to be trained from scratch. To train a model you can use the ``train.py`` script.

### Project Structure
To create a new project, you will need to create the project folder structure inside the ``projects`` directory.
```
projects                  # Projects directory
└── hello_world           # 'hello_world' project files
    ├── frames            # 'hello_world' project video frames
    │   ├── 00000.jpg
    │   ├── 00001.jpg
    │   └── ...
    └── scribbles         # 'hello_world' scribbles files (Only for Multi Layer Foreground)
        ├── 00000.png
        ├── 00001.png
        └── ...
```

### Training
After configuring the file structures for your project, just run the ``train.py`` script and wait for the model to train.
```
python train.py -p hello_world -s 300000
```
- ``-p (--project)``: Select which project will be trained (**Required**)
- ``-s (--steps)``: Sets by how many steps the models will be trained (**Optional**)

This script will generate **Alpha masks** and **Optical flow** files to use in the training process. These files will be saved on ``results`` directory, inside the project's directory.
Here, we have two ways of generating masks for a given video, with ``MaskRCNN`` or with the `MODNet` algorithm.

### MaskRCNN
To use the ``MaskRCNN`` video mask extraction algorithm (**Linux Only**), you can use:
```bash
python train.py -p hello_world -s 300000 -c ./configs/single_layer.json
```
- ``-c (--config)``: Select the configuration file that will be used in the project (**Optional**)

Better results can be achieved by selecting a class for ``MaskRCNN`` through the ``foreground_class`` parameter in a configuration file. For example:
```json
{
  "foreground_class": "bird"
}
```

By default, the ``MaskRCNN`` algorithm will be used for mask extraction with the ``anything`` value in ``foreground_class`` parameter.

**Note:** This repository contains example configuration files in ``./configs`` directory, however custom configuration files can also be used.

### MODNet
You can also use the [MODNet: Trimap-Free Portrait Matting in Real Time](https://github.com/ZHKKKe/MODNet) official implementation
to do video mask extraction. To use it (**Windows Support**), you can use:
```bash
python train.py -p hello_world -s 300000 -c ./configs/single_layer_modnet.json
```

It can also be enabled through the ``foreground_class`` parameter in a configuration file. For example:

```json
{
  "foreground_class": "modnet"
}
```

**Note:** The ``MODNet`` algorithm works better with videos that contains humans.

### Multi Layer Foreground Atlases
The **Multi Layer Foreground Atlases** feature can be enabled through the configuration file `multi_layer.json`:
```bash
python train.py -p hello_world -s 300000 -c ./configs/multi_layer.json
```

Or simply by editing the ``foreground_layers`` property inside any configuration file to an integer value bigger than ``1``.

**Note:** For better results when using the Multi Layer Foreground Atlases feature, you can add
scribbles files to the ``scribbles`` directory of the selected project.

### Evaluating
When a trained model is available, you can run the ``evaluation.py`` script to generate debugging files and visualize the performance of the algorithm on project.
```
python evaluate.py -p hello_world
```
- ``-p (--project)``: Select which project will be evaluated (**Required**)

In most cases, it will not be necessary to run this script, as the evaluation function runs periodically during training.

### Generating Atlases
After training a good model for your project, you can finally start editing the video, starting with the atlas generation process.

By default, the model training process already generates global static atlases for the foreground and background layers that can be used for editing. It can be found at ``./projects/{project_name}/results/atlases``. 

However, with the ``generate_atlases.py`` script, it is possible to generate atlases from edits in a video frame or even enable support for animations.

To generate atlases from an edited video frame, you can do:
```bash
python generate_atlases.py -p hello_world -cf 00000.png
```
- ``-p (--project)``: Select which project to load the models that will be used (**Required**)
- ``-cf (--custom_frame)``: Path to an edited video frame that will be used to generate atlases (**Optional**)

To generate atlases with enabled animation support you only need to add the ``-a`` argument to the command line.
```bash
python generate_atlases.py -p hello_world -a
```
- ``-p (--project)``: Select which project to load the models that will be used (**Required**)
- ``-a (--animated)``: Flag to enable animation support (**Optional**)

**Note:** Unlike static atlases, animated atlases are directories with an atlas file for each frame of the video.

```
animated_atlases          # Animated atlases directory
├── background_atlas      # Background animated atlas
│   ├── 00000.png
│   ├── 00001.png
│   ├── 00002.png
│   ├── 00003.png
│   └── ...
└── foreground_atlas_0    # Foreground animated atlas (Layer 0)
    ├── 00000.png
    ├── 00001.png
    ├── 00002.png
    ├── 00003.png
    └── ...
```

### Rendering Video
And finally, after editing the generated atlases in an image/video editing software (e.g. Photoshop or Premiere),
you can use the ``render_video.py`` script to render the original video with your own edits.

```bash
python render_video.py -p hello_world -f foreground_atlas.png -b background_atlas.png -o output.mp4
```
- ``-p (--project)``: Select which project to load the models that will be used (**Required**)
- ``-f (--fg_atlas)``: Indicates the path of the edited foreground atlas (**Optional**)
- ``-b (--bg_atlas)``: Indicates the path of the edited background atlas (**Optional**)

To use **animated atlases**, just point the foreground/background arguments to the directory of the animated atlas instead of the file:

```bash
python render_video.py -p hello_world -f animated_foreground_atlas -b animated_background_atlas -o output.mp4
```

In some cases, it may be more convenient to use an animated atlas for one layer and a static atlas for another, and this can be done like this:
```bash
python render_video.py -p hello_world -f animated_foreground_atlas -b background_atlas.png -o output.mp4
```

### Removing Foreground / Background
As mentioned in the article, it is also possible to remove the foreground and background from the video by manipulating the **alpha** variable in the pixel reconstruction formula. This effect can be easily achieved by running:
```bash
python render_video.py -r {foreground/background} -o output.mp4
```
- ``-r (--remove)``: Layer to remove from output video. (``foreground`` or ``background``)

**Note:** If neither an atlas file, nor the ``-r (--remove)`` argument is passed to the script ``render_video.py``, it will result in a copy of the original video.

## Citation
If you find their work useful in your research, please consider citing:
```
@article{kasten2021layered,
  title={Layered neural atlases for consistent video editing},
  author={Kasten, Yoni and Ofri, Dolev and Wang, Oliver and Dekel, Tali},
  journal={ACM Transactions on Graphics (TOG)},
  volume={40},
  number={6},
  pages={1--12},
  year={2021},
  publisher={ACM New York, NY, USA}
}
```

## License
NeuralAtlases is released under the MIT license. Please see the LICENSE file for more information.
