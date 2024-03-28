## 

### conf.py

```python
# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.append(os.path.abspath("./_pygments"))

# -- Project information -----------------------------------------------------

project = "nerfstudio"
copyright = "2022, nerfstudio Team"
author = "nerfstudio Team"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxarg.ext",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinxemoji.sphinxemoji",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.mathjax",
    "sphinxext.opengraph",
    "sphinx.ext.viewcode",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
suppress_warnings = ["myst.header"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Needed for interactive plotly in notebooks
html_js_files = [
    "require.min.js",
    "custom.js",
]

# -- MYST configs -----------------------------------------------------------

# To enable admonitions:
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "html_image", "substitution"]


# -- Options for open graph -------------------------------------------------

ogp_site_url = "http://docs.nerf.studio/"
ogp_image = "https://assets.nerf.studio/opg.png"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = "nerfstudio"

autosectionlabel_prefix_document = True

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#d34600",
        "color-brand-content": "#ff6f00",
    },
    "dark_css_variables": {
        "color-brand-primary": "#fdd06c",
        "color-brand-content": "##fea96a",
    },
    "light_logo": "imgs/logo.png",
    "dark_logo": "imgs/logo-dark.png",
}

# -- Code block theme --------------------------------------------------------

pygments_style = "style.NerfstudioStyleLight"
pygments_dark_style = "style.NerfstudioStyleDark"

# -- Napoleon settings -------------------------------------------------------

# Settings for parsing non-sphinx style docstrings. We use Google style in this
# project.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- MYSTNB -----------------------------------------------------------------

suppress_warnings = ["mystnb.unknown_mime_type", "myst.header"]
nb_execution_mode = "off"

```

## 

### index.md

---
myst:
  substitutions:
    luma: |
      ```{image} _static/imgs/luma_light.png
      :alt: Luma AI
      :width: 300px
      :class: only-light
      :target: https://lumalabs.ai/
      ```

      ```{image} _static/imgs/luma_dark.png
      :alt: Luma AI
      :width: 300px
      :class: only-dark
      :target: https://lumalabs.ai/
      ```
    bair: |
      ```{image} _static/imgs/bair_light.png
      :alt: BAIR
      :width: 300px
      :class: only-light
      :target: https://bcommons.berkeley.edu/home
      ```

      ```{image} _static/imgs/bair_dark.png
      :alt: BAIR
      :width: 300px
      :class: only-dark
      :target: https://bcommons.berkeley.edu/home
      ```
---

```{eval-rst}
:og:description: Nerfstudio Documentation
:og:image: https://assets.nerf.studio/opg.png
```

<br/>

```{image} _static/imgs/logo.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-light
```

```{image} _static/imgs/logo-dark.png
:width: 400
:align: center
:alt: nerfstudio
:class: only-dark
```

<br/>

<img src="https://user-images.githubusercontent.com/3310961/194017985-ade69503-9d68-46a2-b518-2db1a012f090.gif" width="52%"/> <img src="https://user-images.githubusercontent.com/3310961/194020648-7e5f380c-15ca-461d-8c1c-20beb586defe.gif" width="46%"/>

<br/>

Nerfstudio provides a simple API that allows for a simplified end-to-end process of creating, training, and testing NeRFs.
The library supports a **more interpretable implementation of NeRFs by modularizing each component.**
With more modular NeRFs, we hope to create a more user-friendly experience in exploring the technology.

This is a contributor-friendly repo with the goal of building a community where users can more easily build upon each other's contributions.
Nerfstudio initially launched as an opensource project by Berkeley students in [KAIR lab](https://people.eecs.berkeley.edu/~kanazawa/index.html#kair) at [Berkeley AI Research (BAIR)](https://bair.berkeley.edu/) in October 2022 as a part of a research project ([paper](https://arxiv.org/abs/2302.04264)). It is currently developed by Berkeley students and community contributors.

We are committed to providing learning resources to help you understand the basics of (if you're just getting started), and keep up-to-date with (if you're a seasoned veteran) all things NeRF. As researchers, we know just how hard it is to get onboarded with this next-gen technology. So we're here to help with tutorials, documentation, and more!

Have feature requests? Want to add your brand-spankin'-new NeRF model? Have a new dataset? **We welcome [contributions](reference/contributing)!**
Please do not hesitate to reach out to the nerfstudio team with any questions via [Discord](https://discord.gg/uMbNqcraFc).

Have feedback? We'd love for you to fill out our [Nerfstudio Feedback Form](https://forms.gle/sqN5phJN7LfQVwnP9) if you want to let us know who you are, why you are interested in Nerfstudio, or provide any feedback!

We hope nerfstudio enables you to build faster 🔨 learn together 📚 and contribute to our NeRF community 💖.

## Contents

```{toctree}
:hidden:
:caption: Getting Started

quickstart/installation
quickstart/first_nerf
quickstart/existing_dataset
quickstart/custom_dataset
quickstart/viewer_quickstart
quickstart/export_geometry
quickstart/data_conventions
Contributing<reference/contributing>
```

```{toctree}
:hidden:
:caption: Extensions
extensions/blender_addon
extensions/unreal_engine
extensions/sdfstudio
```

```{toctree}
:hidden:
:caption: NeRFology

nerfology/methods/index
nerfology/model_components/index
```

```{toctree}
:hidden:
:caption: Developer Guides

developer_guides/new_methods
developer_guides/pipelines/index
developer_guides/viewer/index
developer_guides/config
developer_guides/debugging_tools/index
```

```{toctree}
:hidden:
:caption: Reference

reference/cli/index
reference/api/index
```

This documentation is organized into 3 parts:

- **🏃‍♀️ Getting Started**: a great place to start if you are new to nerfstudio. Contains a quick tour, installation, and an overview of the core structures that will allow you to get up and running with nerfstudio.
- **🧪 Nerfology**: want to learn more about the tech itself? We're here to help with our educational guides. We've provided some interactive notebooks that walk you through what each component is all about.
- **🤓 Developer Guides**: describe all of the components and additional support we provide to help you construct, train, and debug your NeRFs. Learn how to set up a model pipeline, use the viewer, create a custom config, and more.
- **📚 Reference**: describes each class and function. Develop a better understanding of the core of our technology and terminology. This section includes descriptions of each module and component in the codebase.

## Supported Methods

### Included Methods

- [**Nerfacto**](nerfology/methods/nerfacto.md): Recommended method, integrates multiple methods into one.
- [Instant-NGP](nerfology/methods/instant_ngp.md): Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- [NeRF](nerfology/methods/nerf.md): OG Neural Radiance Fields
- [Mip-NeRF](nerfology/methods/mipnerf.md): A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
- [TensoRF](nerfology/methods/tensorf.md): Tensorial Radiance Fields
- [Splatfacto](nerfology/methods/splat.md): Nerfstudio's Gaussian Splatting implementation

(third_party_methods)=

### Third-party Methods

- [Instruct-NeRF2NeRF](nerfology/methods/in2n.md): Editing 3D Scenes with Instructions
- [K-Planes](nerfology/methods/kplanes.md): Unified 3D and 4D Radiance Fields
- [LERF](nerfology/methods/lerf.md): Language Embedded Radiance Fields
- [Nerfbusters](nerfology/methods/nerfbusters.md): Removing Ghostly Artifacts from Casually Captured NeRFs
- [NeRFPlayer](nerfology/methods/nerfplayer.md): 4D Radiance Fields by Streaming Feature Channels
- [Tetra-NeRF](nerfology/methods/tetranerf.md): Representing Neural Radiance Fields Using Tetrahedra
- [Instruct-GS2GS](nerfology/methods/igs2gs.md): Editing 3DGS Scenes with Instructions
- [PyNeRF](nerfology/methods/pynerf.md): Pyramidal Neural Radiance Fields
- [SeaThru-NeRF](nerfology/methods/seathru_nerf.md): Neural Radiance Field for subsea scenes
- [Zip-NeRF](nerfology/methods/zipnerf.md): Anti-Aliased Grid-Based Neural Radiance Fields

**Eager to contribute a method?** We'd love to see you use nerfstudio in implementing new (or even existing) methods! Please view our {ref}`guide<own_method_docs>` for more details about how to add to this list!

## Quicklinks

|                                                            |                        |
| ---------------------------------------------------------- | ---------------------- |
| [Github](https://github.com/nerfstudio-project/nerfstudio) | Official Github Repo   |
| [Discord](https://discord.gg/RyVk6w5WWP)                   | Join Discord Community |
| [Feedback Form](https://forms.gle/sqN5phJN7LfQVwnP9)       | Provide Nerfstudio Feedback |

## Sponsors
Sponsors of this work includes [Luma AI](https://lumalabs.ai/) and the [BAIR commons](https://bcommons.berkeley.edu/home).

|          |          |
| -------- | -------- |
| {{luma}} | {{bair}} |

## Built On

```{image} https://brentyi.github.io/tyro/_static/logo-light.svg
:width: 150
:alt: tyro
:class: only-light
:target: https://github.com/brentyi/tyro
```

```{image} https://brentyi.github.io/tyro/_static/logo-dark.svg
:width: 150
:alt: tyro
:class: only-dark
:target: https://github.com/brentyi/tyro
```

- Easy to use config system
- Developed by [Brent Yi](https://brentyi.com/)

```{image} https://user-images.githubusercontent.com/3310961/199084143-0d63eb40-3f35-48d2-a9d5-78d1d60b7d66.png
:width: 250
:alt: tyro
:class: only-light
:target: https://github.com/KAIR-BAIR/nerfacc
```

```{image} https://user-images.githubusercontent.com/3310961/199083722-881a2372-62c1-4255-8521-31a95a721851.png
:width: 250
:alt: tyro
:class: only-dark
:target: https://github.com/KAIR-BAIR/nerfacc
```

- Library for accelerating NeRF renders
- Developed by [Ruilong Li](https://www.liruilong.cn/)

## Citation

You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2302.04264).

If you use this library or find the documentation useful for your research, please consider citing:

```none
@inproceedings{nerfstudio,
	title        = {Nerfstudio: A Modular Framework for Neural Radiance Field Development},
	author       = {
		Tancik, Matthew and Weber, Ethan and Ng, Evonne and Li, Ruilong and Yi, Brent
		and Kerr, Justin and Wang, Terrance and Kristoffersen, Alexander and Austin,
		Jake and Salahi, Kamyar and Ahuja, Abhik and McAllister, David and Kanazawa,
		Angjoo
	},
	year         = 2023,
	booktitle    = {ACM SIGGRAPH 2023 Conference Proceedings},
	series       = {SIGGRAPH '23}
}
```

## Contributors

<a href="https://github.com/nerfstudio-project/nerfstudio/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=nerfstudio-project/nerfstudio" />
</a>

## Maintainers

|                                                 | Nerfstudio Discord | Affiliation                          |
| ----------------------------------------------- | ------------------ | ------------------------------------ |
| [Justin Kerr](https://kerrj.github.io/)         | justin.kerr        | UC Berkeley                          |
| [Jonáš Kulhánek](https://jkulhanek.com/)        | jkulhanek          | Czech Technical University in Prague |
| [Matt Tancik](https://www.matthewtancik.com)    | tancik             | Luma AI                              |
| [Matias Turkulainen](https://maturk.github.io/) | maturk             | ETH Zurich                           |
| [Ethan Weber](https://ethanweber.me/)           | ethanweber         | UC Berkeley                          |
| [Brent Yi](https://github.com/brentyi)          | brent              | UC Berkeley                          |
## _pygments

### style.py

```python
"""Custom Pygments styles for the Sphinx documentation."""

from pygments.style import Style
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
    Whitespace,
)


class NerfstudioStyleLight(Style):
    """
    A style based on the manni pygments style.
    """

    background_color = "#f8f9fb"

    styles = {
        Whitespace: "#bbbbbb",
        Comment: "italic #d34600",
        Comment.Preproc: "noitalic #009999",
        Comment.Special: "bold",
        Keyword: "bold #006699",
        Keyword.Pseudo: "nobold",
        Keyword.Type: "#007788",
        Operator: "#555555",
        Operator.Word: "bold #000000",
        Name.Builtin: "#336666",
        Name.Function: "#CC00FF",
        Name.Class: "bold #00AA88",
        Name.Namespace: "bold #00CCFF",
        Name.Exception: "bold #CC0000",
        Name.Variable: "#003333",
        Name.Constant: "#336600",
        Name.Label: "#9999FF",
        Name.Entity: "bold #999999",
        Name.Attribute: "#330099",
        Name.Tag: "bold #330099",
        Name.Decorator: "#9999FF",
        String: "#CC3300",
        String.Doc: "italic",
        String.Interpol: "#AA0000",
        String.Escape: "bold #CC3300",
        String.Regex: "#33AAAA",
        String.Symbol: "#FFCC33",
        String.Other: "#CC3300",
        Number: "#FF6600",
        Generic.Heading: "bold #003300",
        Generic.Subheading: "bold #003300",
        Generic.Deleted: "border:#CC0000 bg:#FFCCCC",
        Generic.Inserted: "border:#00CC00 bg:#CCFFCC",
        Generic.Error: "#FF0000",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Generic.Prompt: "bold #000099",
        Generic.Output: "#AAAAAA",
        Generic.Traceback: "#99CC66",
        Error: "bg:#FFAAAA #AA0000",
    }


class NerfstudioStyleDark(Style):
    """
    A style based on the one-dark style.
    """

    background_color = "#282C34"

    styles = {
        Token: "#ABB2BF",
        Punctuation: "#ABB2BF",
        Punctuation.Marker: "#ABB2BF",
        Keyword: "#C678DD",
        Keyword.Constant: "#fdd06c",
        Keyword.Declaration: "#C678DD",
        Keyword.Namespace: "#C678DD",
        Keyword.Reserved: "#C678DD",
        Keyword.Type: "#fdd06c",
        Name: "#ff8c58",
        Name.Attribute: "#ff8c58",
        Name.Builtin: "#fdd06c",
        Name.Class: "#fdd06c",
        Name.Function: "bold #61AFEF",
        Name.Function.Magic: "bold #56B6C2",
        Name.Other: "#ff8c58",
        Name.Tag: "#ff8c58",
        Name.Decorator: "#61AFEF",
        Name.Variable.Class: "",
        String: "#bde3a1",
        Number: "#D19A66",
        Operator: "#56B6C2",
        Comment: "#7F848E",
    }

```

## quickstart

### first_nerf.md

# Training your first model

## Train and run viewer

The following will train a _nerfacto_ model, our recommended model for real world scenes.

```bash
# Download some test data:
ns-download-data nerfstudio --capture-name=poster
# Train model
ns-train nerfacto --data data/nerfstudio/poster
```

If everything works, you should see training progress like the following:

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766069-cadfd34f-8833-4156-88b7-ad406d688fc0.png">
</p>

Navigating to the link at the end of the terminal will load the webviewer. If you are running on a remote machine, you will need to port forward the websocket port (defaults to 7007).

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/3310961/202766653-586a0daa-466b-4140-a136-6b02f2ce2c54.png">
</p>

:::{admonition} Note
:class: note

- You may have to change the port using `--viewer.websocket-port`.
- All data configurations must go at the end. In this case, `nerfstudio-data` and all of its corresponding configurations come at the end after the model and viewer specification.
  :::

## Resume from checkpoint

It is possible to load a pretrained model by running

```bash
ns-train nerfacto --data data/nerfstudio/poster --load-dir {outputs/.../nerfstudio_models}
```

## Visualize existing run

Given a pretrained model checkpoint, you can start the viewer by running

```bash
ns-viewer --load-config {outputs/.../config.yml}
```

## Exporting Results

Once you have a NeRF model you can either render out a video or export a point cloud.

### Render Video

First we must create a path for the camera to follow. This can be done in the viewer under the "RENDER" tab. Orient your 3D view to the location where you wish the video to start, then press "ADD CAMERA". This will set the first camera key frame. Continue to new viewpoints adding additional cameras to create the camera path. We provide other parameters to further refine your camera path. Once satisfied, press "RENDER" which will display a modal that contains the command needed to render the video. Kill the training job (or create a new terminal if you have lots of compute) and run the command to generate the video.

Other video export options are available, learn more by running

```bash
ns-render --help
```

### Generate Point Cloud

While NeRF models are not designed to generate point clouds, it is still possible. Navigate to the "EXPORT" tab in the 3D viewer and select "POINT CLOUD". If the crop option is selected, everything in the yellow square will be exported into a point cloud. Modify the settings as desired then run the command at the bottom of the panel in your command line.

Alternatively you can use the CLI without the viewer. Learn about the export options by running

```bash
ns-export pointcloud --help
```

## Intro to nerfstudio CLI and Configs

Nerfstudio allows customization of training and eval configs from the CLI in a powerful way, but there are some things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following commands:

The following will list the supported models

```bash
ns-train --help
```

Applying `--help` after the model specification will provide the model and training specific arguments.

```bash
ns-train nerfacto --help
```

At the end of the command you can specify the dataparser used. By default we use the _nerfstudio-data_ dataparser. We include other dataparsers such as _Blender_, _NuScenes_, ect. For a list of dataparse specific arguments, add `--help` to the end of the command

```bash
ns-train nerfacto <nerfacto optional args> nerfstudio-data --help
```

Each script will have some other minor quirks (like the training script dataparser subcommand needing to come after the model subcommand), read up on them [here](../reference/cli/index.md).

## Comet / Tensorboard / WandB / Viewer

We support four different methods to track training progress, using the viewer [tensorboard](https://www.tensorflow.org/tensorboard), [Weights and Biases](https://wandb.ai/site), and [Comet](https://comet.com/?utm_source=nerf&utm_medium=referral&utm_content=nerf_docs). You can specify which visualizer to use by appending `--vis {viewer, tensorboard, wandb, comet, viewer+wandb, viewer+tensorboard, viewer+comet}` to the training command. Simultaneously utilizing the viewer alongside wandb or tensorboard may cause stuttering issues during evaluation steps. The viewer only works for methods that are fast (ie. nerfacto, instant-ngp), for slower methods like NeRF, use the other loggers.

## Evaluating Runs

Calculate the psnr of your trained model and save to a json file.

```bash
ns-eval --load-config={PATH_TO_CONFIG} --output-path=output.json
```

We also provide a train and evaluation script that allows you to do benchmarking on the classical Blender dataset (see our [benchmarking workflow](../developer_guides/debugging_tools/benchmarking.md)).

## Multi-GPU Training

Here we explain how to use multi-GPU training. We are using [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/beginner/dist_overview.html), so gradients are averaged over devices. If the loss scales depend on sample size (usually not the case in our implementation), we need to scale the learning rate with the number of GPUs used. Plotting will only be done for the first process. Note that you may want to play around with both learning rate and `<X>_num_rays_per_batch` when using DDP. Below is a simple example for how you'd run the `nerfacto-big` method on the aspen scene (see above to download the data), with either 1 or 2 GPUs. The `nerfacto-big` method uses a larger model size than the `nerfacto` method, so it benefits more from multi-GPU training.

First, download the aspen scene.

```python
ns-download-data nerfstudio --capture-name=aspen
```

```python
# 1 GPU (8192 rays per GPU per batch)
export CUDA_VISIBLE_DEVICES=0
ns-train nerfacto-big --vis viewer+wandb --machine.num-devices 1 --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen
```

You would observe about ~70k rays / sec on NVIDIA V100.

```
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec
-----------------------------------------------------------------------------------
610 (0.61%)         115.968 ms           3 h, 12 m, 6 s       72.68 K
620 (0.62%)         115.908 ms           3 h, 11 m, 58 s      72.72 K
630 (0.63%)         115.907 ms           3 h, 11 m, 57 s      72.73 K
640 (0.64%)         115.937 ms           3 h, 11 m, 59 s      72.71 K
650 (0.65%)         115.853 ms           3 h, 11 m, 49 s      72.76 K
660 (0.66%)         115.710 ms           3 h, 11 m, 34 s      72.85 K
670 (0.67%)         115.797 ms           3 h, 11 m, 42 s      72.80 K
680 (0.68%)         115.783 ms           3 h, 11 m, 39 s      72.81 K
690 (0.69%)         115.756 ms           3 h, 11 m, 35 s      72.81 K
700 (0.70%)         115.755 ms           3 h, 11 m, 34 s      72.81 K
```

By having more GPUs in the training, you can allocate batch size to multiple GPUs and average their gradients.

```python
# 2 GPUs (4096 rays per GPU per batch, effectively 8192 rays per batch)
export CUDA_VISIBLE_DEVICES=0,1
ns-train nerfacto --vis viewer+wandb --machine.num-devices 2 --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen
```

You would get improved throughput (~100k rays / sec on two NVIDIA V100).

```
Step (% Done)       Train Iter (time)    ETA (time)           Train Rays / Sec
-----------------------------------------------------------------------------------
1910 (1.91%)        79.623 ms            2 h, 10 m, 10 s      104.92 K
1920 (1.92%)        79.083 ms            2 h, 9 m, 16 s       105.49 K
1930 (1.93%)        79.092 ms            2 h, 9 m, 16 s       105.48 K
1940 (1.94%)        79.364 ms            2 h, 9 m, 42 s       105.21 K
1950 (1.95%)        79.327 ms            2 h, 9 m, 38 s       105.25 K
1960 (1.96%)        79.473 ms            2 h, 9 m, 51 s       105.09 K
1970 (1.97%)        79.334 ms            2 h, 9 m, 37 s       105.26 K
1980 (1.98%)        79.200 ms            2 h, 9 m, 23 s       105.38 K
1990 (1.99%)        79.264 ms            2 h, 9 m, 28 s       105.29 K
2000 (2.00%)        79.168 ms            2 h, 9 m, 18 s       105.40 K
```

During training, the "Train Rays / Sec" throughput represents the total number of training rays it processes per second, gradually increase the number of GPUs and observe how this throughput improves and eventually saturates.
## quickstart

### existing_dataset.md

# Using existing data

Nerfstudio comes with built-in support for a number of datasets, which can be downloaded with the [`ns-download-data` command][cli]. Each of the built-in datasets comes ready to use with various Nerfstudio methods (e.g. the recommended default Nerfacto), allowing you to get started in the blink of an eye.

[cli]: https://docs.nerf.studio/reference/cli/ns_download_data.html
[paper]: https://arxiv.org/pdf/2302.04264.pdf

## Example

Here are a few examples of downloading different scenes. Please see the [Training Your First NeRF](first_nerf.md) documentation for more details on how to train a model with them.

```bash
# Download all scenes from the Blender dataset, including the "classic" Lego model
ns-download-data blender

# Download the subset of data used in the SIGGRAPH 2023 Nerfstudio paper
ns-download-data nerfstudio --capture-name nerfstudio-dataset

# Download a few room-scale scenes from the EyefulTower dataset at different resolutions
ns-download-data eyefultower --capture-name riverview seating_area apartment --resolution-name jpeg_1k jpeg_2k

# Download the full D-NeRF dataset of dynamic synthetic scenes
ns-download-data dnerf
```

## Dataset Summary

Many of these datasets are used as baselines to evaluate new research in novel view synthesis, such as in the [original Nerfstudio paper][paper]. Scenes from these datasets lie at dramatically different points in the space of images, across axes such as photorealism (synthetic vs real), dynamic range (LDR vs HDR), scale (number of images), and resolution. The tables below describe some of this variation, and hopefully make it easier to pick an appropriate dataset for your research or application.

| Dataset | Synthetic | Real | LDR | HDR | Scenes | Image Count<sup>1</sup> | Image Resolution<sup>2</sup> |
| :-: | :-: | :-: | :-: | :-: | :------: | :-: | :-: |
| [Blender][blender] | ✔️ |  | ✔️ |  | 8 | ➖➕️➖➖ | ➕️➖➖➖➖ |
| [D-NeRF][dnerf] | ✔️ |  | ✔️ |  | 8 | ➕️➖➖➖ | ➕️➖➖➖➖ |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ | 11 | ➖➕️➕️➕️ | ➖➕️➕️➕️➕️ |
| [Mill 19][mill19] |  | ✔️ | ✔️ |  | 2 | ➖➖➕️➖ | ➖➖➖➕️➖ |
| [NeRF-OSR][nerfosr] |  | ✔️ | ✔️ |  | 9 | ➕➕️➕️➖ | ➖➕️➖➕️➖ |
| [Nerfstudio][nerfstudio] |  | ✔️ | ✔️ |  | 18 | ➕➕️➕️➖ | ➕️➕️➕️➖➖ |
| [PhotoTourism][phototourism] |  | ✔️ | ✔️ |  | 10 | ➖➕️➕️➖ | ➖➕️➖➖➖ |
| [Record3D][record3d] |  | ✔️ | ✔️ |  | 1 | ➖➖➕️➖ | ➕️➖➖➖➖ |
| [SDFStudio][sdfstudio] | ✔️ | ✔️ | ✔️ |  | 45 | ➕️➕️➕️➖ | ➕️➖➕️➖➖ |
| [sitcoms3D][sitcoms3d] |  | ✔️ | ✔️ |  | 10 | ➕️➖➖➖ | ➕️➕️➖➖➖ |

In the tables below, each dataset was placed into a bucket based on the table's chosen property. If a box contains a ✔️, the corresponding dataset will have *at least* one scene falling into the corresponding bucket for that property, though there may be multiple scenes at different points within the range.

<sub>
<b>1:</b> Condensed version of the "Scene Size: Number of RGB Images" table below. <br>
<b>2:</b> Condensed version of the "Scene RGB Resolutions: `max(width, height)`" table below.
</sub>

### Scene Size: Number of RGB Images

| Dataset | < 250 | 250 - 999 | 1000 - 3999 | ≥ 4000 |
| :-: | :-: | :-: | :-: | :-: |
| [Blender][blender] |  | ✔️ |  |  |
| [D-NeRF][dnerf] | ✔️ |  |  |  |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ |
| [Mill 19][mill19] |  |  | ✔️ |  |
| [NeRF-OSR][nerfosr] | ✔️ | ✔️ | ✔️ |  |
| [Nerfstudio][nerfstudio] | ✔️ | ✔️ | ✔️ |  |
| [PhotoTourism][phototourism] |  | ✔️ | ✔️ |  |
| [Record3D][record3d] |  |  | ✔️ |  |
| [SDFStudio][sdfstudio] | ✔️ | ✔️ | ✔️ |  |
| [sitcoms3D][sitcoms3d] | ✔️ |  |  |

### Scene RGB Resolutions: `max(width, height)`

| Dataset | < 1000 | 1000 - 1999 | 2000 - 3999 | 4000 - 7999 | ≥ 8000 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| [Blender][blender] | ✔️ |  |  |  |  |
| [D-NeRF][dnerf] | ✔️ |  |  |  |  |
| [EyefulTower][eyefultower] |  | ✔️ | ✔️ | ✔️ | ✔️ |
| [Mill 19][mill19] |  |  |  | ✔️ |  |
| [NeRF-OSR][nerfosr] |  | ✔️ |  | ✔️ |  |
| [Nerfstudio][nerfstudio] | ✔️ | ✔️ | ✔️ |  |  |
| [PhotoTourism][phototourism] |  | ✔️ |  |  |  |
| [Record3D][record3d] | ✔️ |  |  |  |  |
| [SDFStudio][sdfstudio] | ✔️ |  | ✔️ |  |  |
| [sitcoms3D][sitcoms3d] | ✔️ | ✔️ |  |  |  |

[blender]: https://github.com/bmild/nerf?tab=readme-ov-file#project-page--video--paper--data
[dnerf]: https://github.com/albertpumarola/D-NeRF?tab=readme-ov-file#download-dataset
[eyefultower]: https://github.com/facebookresearch/EyefulTower
[mill19]: https://github.com/cmusatyalab/mega-nerf?tab=readme-ov-file#mill-19
[nerfosr]: https://4dqv.mpi-inf.mpg.de/NeRF-OSR/
[nerfstudio]: https://github.com/nerfstudio-project/nerfstudio
[phototourism]: https://www.cs.ubc.ca/~kmyi/imw2020/data.html
[record3d]: https://record3d.app/
[sdfstudio]: https://github.com/autonomousvision/sdfstudio/blob/master/docs/sdfstudio-data.md#Existing-dataset
[sitcoms3d]: https://github.com/ethanweber/sitcoms3D/blob/master/METADATA.md
## quickstart

### data_conventions.md

# Data conventions

## Coordinate conventions

Here we explain the coordinate conventions for using our repo.

### Camera/view space

We use the OpenGL/Blender (and original NeRF) coordinate convention for cameras. +X is right, +Y is up, and +Z is pointing back and away from the camera. -Z is the look-at direction. Other codebases may use the COLMAP/OpenCV convention, where the Y and Z axes are flipped from ours but the +X axis remains the same.

### World space

Our world space is oriented such that the up vector is +Z. The XY plane is parallel to the ground plane. In the viewer, you'll notice that red, green, and blue vectors correspond to X, Y, and Z respectively.

<hr>

## Dataset format

Our explanation here is for the nerfstudio data format. The `transforms.json` has a similar format to [Instant NGP](https://github.com/NVlabs/instant-ngp).

### Camera intrinsics

If all of the images share the same camera intrinsics, the values can be placed at the top of the file.

```json
{
  "camera_model": "OPENCV_FISHEYE", // camera model type [OPENCV, OPENCV_FISHEYE]
  "fl_x": 1072.0, // focal length x
  "fl_y": 1068.0, // focal length y
  "cx": 1504.0, // principal point x
  "cy": 1000.0, // principal point y
  "w": 3008, // image width
  "h": 2000, // image height
  "k1": 0.0312, // first radial distortion parameter, used by [OPENCV, OPENCV_FISHEYE]
  "k2": 0.0051, // second radial distortion parameter, used by [OPENCV, OPENCV_FISHEYE]
  "k3": 0.0006, // third radial distortion parameter, used by [OPENCV_FISHEYE]
  "k4": 0.0001, // fourth radial distortion parameter, used by [OPENCV_FISHEYE]
  "p1": -6.47e-5, // first tangential distortion parameter, used by [OPENCV]
  "p2": -1.37e-7, // second tangential distortion parameter, used by [OPENCV]
  "frames": // ... per-frame intrinsics and extrinsics parameters
}
```

Per-frame intrinsics can also be defined in the `frames` field. If defined for a field (ie. `fl_x`), all images must have per-image intrinsics defined for that field. Per-frame `camera_model` is not supported.

```json
{
  // ...
  "frames": [
    {
      "fl_x": 1234
    }
  ]
}
```

### Camera extrinsics

For a transform matrix, the first 3 columns are the +X, +Y, and +Z defining the camera orientation, and the X, Y, Z values define the origin. The last row is to be compatible with homogeneous coordinates.

```json
{
  // ...
  "frames": [
    {
      "file_path": "images/frame_00001.jpeg",
      "transform_matrix": [
        // [+X0 +Y0 +Z0 X]
        // [+X1 +Y1 +Z1 Y]
        // [+X2 +Y2 +Z2 Z]
        // [0.0 0.0 0.0 1]
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
      // Additional per-frame info
    }
  ]
}
```

### Depth images

To train with depth supervision, you can also provide a `depth_file_path` for each frame in your `transforms.json` and use one of the methods that support additional depth losses (e.g., depth-nerfacto). The depths are assumed to be 16-bit or 32-bit and to be in millimeters to remain consistent with [Polyform](https://github.com/PolyCam/polyform). Zero-value in the depth image is treated as unknown depth. You can adjust this scaling factor using the `depth_unit_scale_factor` parameter in `NerfstudioDataParserConfig`. Note that by default, we resize the depth images to match the shape of the RGB images.

```json
{
  "frames": [
    {
      // ...
      "depth_file_path": "depth/0001.png"
    }
  ]
}
```

### Masks

:::{admonition} Warning
:class: Warning

The current implementation of masking is inefficient and will cause large memory allocations.
:::

There may be parts of the training image that should not be used during training (ie. moving objects such as people). These images can be masked out using an additional mask image that is specified in the `frame` data.

```json
{
  "frames": [
    {
      // ...
      "mask_path": "masks/mask.jpeg"
    }
  ]
}
```

The following mask requirements must be met:

- Must be 1 channel with only black and white pixels
- Must be the same resolution as the training image
- Black corresponds to regions to ignore
- If used, all images must have a mask
## quickstart

### custom_dataset.md

# Using custom data

Training model on existing datasets is only so fun. If you would like to train on self captured data you will need to process the data into the nerfstudio format. Specifically we need to know the camera poses for each image.

To process your own data run:

```bash
ns-process-data {video,images,polycam,record3d} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
```

A full set of arguments can be found {doc}`here</reference/cli/ns_process_data>`.

We currently support the following custom data types:
| Data | Capture Device | Requirements | `ns-process-data` Speed |
| ----------------------------- | -------------- | ----------------------------------------------- | ----------------------- |
| 📷 [Images](images_and_video) | Any | [COLMAP](https://colmap.github.io/install.html) | 🐢 |
| 📹 [Video](images_and_video) | Any | [COLMAP](https://colmap.github.io/install.html) | 🐢 |
| 🌎 [360 Data](360_data) | Any | [COLMAP](https://colmap.github.io/install.html) | 🐢 |
| 📱 [Polycam](polycam) | IOS with LiDAR | [Polycam App](https://poly.cam/) | 🐇 |
| 📱 [KIRI Engine](kiri) | IOS or Android | [KIRI Engine App](https://www.kiriengine.com/) | 🐇 |
| 📱 [Record3D](record3d) | IOS with LiDAR | [Record3D app](https://record3d.app/) | 🐇 |
| 📱 [Spectacular AI](spectacularai) | IOS, OAK, others| [App](https://apps.apple.com/us/app/spectacular-rec/id6473188128) / [`sai-cli`](https://www.spectacularai.com/mapping) | 🐇 |
| 🖥 [Metashape](metashape) | Any | [Metashape](https://www.agisoft.com/) | 🐇 |
| 🖥 [RealityCapture](realitycapture) | Any | [RealityCapture](https://www.capturingreality.com/realitycapture) | 🐇 |
| 🖥 [ODM](odm) | Any | [ODM](https://github.com/OpenDroneMap/ODM) | 🐇 |
| 👓 [Aria](aria) | Aria glasses | [Project Aria](https://projectaria.com/) | 🐇 |

(images_and_video)=

## Images or Video

To assist running on custom data we have a script that will process a video or folder of images into a format that is compatible with nerfstudio. We use [COLMAP](https://colmap.github.io) and [FFmpeg](https://ffmpeg.org/download.html) in our data processing script, please have these installed. We have provided a quickstart to installing COLMAP below, FFmpeg can be downloaded from [here](https://ffmpeg.org/download.html)

:::{admonition} Tip
:class: info

- COLMAP can be finicky. Try your best to capture overlapping, non-blurry images.
  :::

### Processing Data

```bash
ns-process-data {images, video} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
```

### Training on your data

```bash
ns-train nerfacto --data {PROCESSED_DATA_DIR}
```

### Training and evaluation on separate data

For `ns-process-data {images, video}`, you can optionally use a separate image directory or video for training and evaluation, as suggested in [Nerfbusters](https://ethanweber.me/nerfbusters/). To do this, run `ns-process-data {images, video} --data {DATA_PATH} --eval-data {EVAL_DATA_PATH} --output-dir {PROCESSED_DATA_DIR}`. Then when running nerfacto, run `ns-train nerfacto --data {PROCESSED_DATA_DIR} nerfstudio-data --eval-mode filename`.

### Installing COLMAP

There are many ways to install COLMAP, unfortunately it can sometimes be a bit finicky. If the following commands do not work, please refer to the [COLMAP installation guide](https://colmap.github.io/install.html) for additional installation methods. COLMAP install issues are common! Feel free to ask for help in on our [Discord](https://discord.gg/uMbNqcraFc).

::::::{tab-set}
:::::{tab-item} Linux

We recommend trying `conda`:

```
conda install -c conda-forge colmap
```

Check that COLMAP 3.8 with CUDA is successfully installed:

```
colmap -h
```

If that doesn't work, you can try VKPG:
::::{tab-set}
:::{tab-item} CUDA

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap[cuda]:x64-linux
```

:::
:::{tab-item} CPU

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap:x64-linux
```

:::
::::

If that doesn't work, you will need to build from source. Refer to the [COLMAP installation guide](https://colmap.github.io/install.html) for details.

:::::

:::::{tab-item} OSX

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg install colmap
```

:::::

:::::{tab-item} Windows

::::{tab-set}
:::{tab-item} CUDA

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install colmap[cuda]:x64-windows
```

:::
:::{tab-item} CPU

```bash
git clone https://github.com/microsoft/vcpkg
cd vcpkg
.\bootstrap-vcpkg.sh
.\vcpkg install colmap:x64-windows
```

:::
::::

:::::
::::::

(polycam)=

## Polycam Capture

Nerfstudio can also be trained directly from captures from the [Polycam app](https://poly.cam//). This avoids the need to use COLMAP. Polycam's poses are globally optimized which make them more robust to drift (an issue with ARKit or SLAM methods).

To get the best results, try to reduce motion blur as much as possible and try to view the target from as many viewpoints as possible. Polycam recommends having good lighting and moving the camera slowly if using auto mode. Or, even better, use the manual shutter mode to capture less blurry images.

:::{admonition} Note
:class: info
A LiDAR enabled iPhone or iPad is necessary.
:::

### Setting up Polycam

```{image} imgs/polycam_settings.png
:width: 200
:align: center
:alt: polycam settings
```

Developer settings must be enabled in Polycam. To do this, navigate to the settings screen and select `Developer mode`. Note that this will only apply for future captures, you will not be able to process existing captures with nerfstudio.

### Process data

```{image} imgs/polycam_export.png
:width: 400
:align: center
:alt: polycam export options
```

0. Capture data in LiDAR or Room mode.

1. Tap `Process` to process the data in the Polycam app.

2. Navigate to the export app pane.

3. Select `raw data` to export a `.zip` file.

4. Convert the Polycam data into the nerfstudio format using the following command:

```bash
ns-process-data polycam --data {OUTPUT_FILE.zip} --output-dir {output directory}
```

5. Train with nerfstudio!

```bash
ns-train nerfacto --data {output directory}
```

(kiri)=

## KIRI Engine Capture

Nerfstudio can trained from data processed by the [KIRI Engine app](https://www.kiriengine.com/). This works for both Android and iPhone and does not require a LiDAR supported device.

:::{admonition} Note
:class: info
`ns-process-data` does not need to be run when using KIRI Engine.
:::

### Setting up KIRI Engine

```{image} imgs/kiri_setup.png
:width: 400
:align: center
:alt: KIRI Engine setup
```

After downloading the app, `Developer Mode` needs to be enabled. A toggle can be found in the settings menu.

### Process data

```{image} imgs/kiri_capture.png
:width: 400
:align: center
:alt: KIRI Engine setup
```

1. Navigate to captures window.

2. Select `Dev.` tab.

3. Tap the `+` button to create a new capture.

4. Choose `Camera pose` as the capture option.

5. Capture the scene and provide a name.

6. After processing is complete, export the scene. It will be sent to your email.

7. Unzip the file and run the training script (`ns-process-data` is not necessary).

```bash
ns-train nerfacto --data {kiri output directory}
```

(record3d)=

## Record3D Capture

Nerfstudio can be trained directly from >=iPhone 12 Pro captures from the [Record3D app](https://record3d.app/). This uses the iPhone's LiDAR sensors to calculate camera poses, so COLMAP is not needed.

Click on the image down below 👇 for a 1-minute tutorial on how to run nerfstudio with Record3D from start to finish.

[![How to easily use nerfstudio with Record3D](imgs/record3d_promo.png)](https://youtu.be/XwKq7qDQCQk 'How to easily use nerfstudio with Record3D')

At a high level, you can follow these 3 steps:

1. Record a video and export with the EXR + JPG sequence format.

  <img src="imgs/record_3d_video_selection.png" width=150>
  <img src="imgs/record_3d_export_selection.png" width=150>

2. Then, move the exported capture folder from your iPhone to your computer.

3. Convert the data to the nerfstudio format.

```bash
ns-process-data record3d --data {data directory} --output-dir {output directory}
```

4. Train with nerfstudio!

```bash
ns-train nerfacto --data {output directory}
```

(spectacularai)=

## Spectacular AI

Spectacular AI SDK and apps can be used to capture data from various devices:

 * iPhones (with LiDAR)
 * OAK-D cameras
 * RealSense D455/D435i
 * Azure Kinect DK

The SDK also records IMU data, which is fused with camera and (if available) LiDAR/ToF data when computing the camera poses. This approach, VISLAM, is more robust than purely image based methods (e.g., COLMAP) and can work better and faster for difficult data (monotonic environments, fast motions, narrow FoV, etc.).

Instructions:

1. Installation. With the Nerfstudio Conda environment active, first install the Spectacular AI Python library

```bash
pip install spectacularAI[full]
```

2. Install FFmpeg. Linux: `apt install ffmpeg` (or similar, if using another package manager). Windows: [see here](https://www.editframe.com/guides/how-to-install-and-start-using-ffmpeg-in-under-10-minutes). FFmpeg must be in your `PATH` so that `ffmpeg` works on the command line.

3. Data capture. See [here for specific instructions for each supported device](https://github.com/SpectacularAI/sdk-examples/tree/main/python/mapping#recording-data).
  
4. Process and export. Once you have recorded a dataset in Spectacular AI format and have it stored in `{data directory}` it can be converted into a Nerfstudio supported format with:

```bash
sai-cli process {data directory} --preview3d --key_frame_distance=0.05 {output directory}
```
The optional `--preview3d` flag shows a 3D preview of the point cloud and estimated trajectory live while VISLAM is running. The `--key_frame_distance` argument can be tuned based on the recorded scene size: 0.05 (5cm) is good for small scans and 0.15 for room-sized scans. If the processing gets slow, you can also try adding a --fast flag to `sai-cli process` to trade off quality for speed. 

5. Train. No separate `ns-process-data` step is needed. The data in `{output directory}` can now be trained with Nerfstudio:

```bash
ns-train nerfacto --data {output directory}
```

(metashape)=

## Metashape

All images must use the same sensor type (but multiple sensors are supported).

1. Align your images using Metashape. `File -> Workflow -> Align Photos...`

```{image} https://user-images.githubusercontent.com/3310961/203389662-12760210-2b52-49d4-ab21-4f23bfa4a2b3.png
:width: 400
:align: center
:alt: metashape alignment
```

2. Export the camera alignment as a `xml` file. `File -> Export -> Export Cameras...`

```{image} https://user-images.githubusercontent.com/3310961/203385691-74565704-e4f6-4034-867e-5d8b940fc658.png
:width: 400
:align: center
:alt: metashape export
```

3. Convert the data to the nerfstudio format.

```bash
ns-process-data metashape --data {data directory} --xml {xml file} --output-dir {output directory}
```

4. Train with nerfstudio!

```bash
ns-train nerfacto --data {output directory}
```

(realitycapture)=

## RealityCapture

1. Align your images using RealityCapture. `ALIGNMENT -> Align Images`

2. Export the camera alignment as a `csv` file. Choose `Internal/External camera parameters`

3. Convert the data to the nerfstudio format.

```bash
ns-process-data realitycapture --data {data directory} --csv {csv file} --output-dir {output directory}
```

4. Train with nerfstudio!

```bash
ns-train nerfacto --data {output directory}
```

(odm)=

## ODM

All images/videos must be captured with the same camera.

1. Process a dataset using [ODM](https://github.com/OpenDroneMap/ODM#quickstart)

```bash
$ ls /path/to/dataset
images
odm_report
odm_orthophoto
...
```

2. Convert to nerfstudio format.

```bash
ns-process-data odm --data /path/to/dataset --output-dir {output directory}
```

4. Train!

```bash
ns-train nerfacto --data {output directory}
```

(aria)=

## Aria

1. Install projectaria_tools:

```bash
conda activate nerfstudio
pip install projectaria-tools'[all]'
```

2. Download a VRS file from Project Aria glasses, and run Machine Perception Services to extract poses.

3. Convert to nerfstudio format.

```bash
ns-process-data aria --vrs-file /path/to/vrs/file --mps-data-dir /path/to/mps/data --output-dir {output directory}
```

4. Train!

```bash
ns-train nerfacto --data {output directory}
```

(360_data)=

## 360 Data (Equirectangular)

Equirectangular data is data that has been taken by a 360 camera such as Insta360. Both equirectangular image sets and videos can be processed by nerfstudio.

### Images

For a set of equirectangular images, process the data using the following command:

```bash
ns-process-data images --camera-type equirectangular --images-per-equirect {8, or 14} --crop-factor {top bottom left right} --data {data directory} --output-dir {output directory}
```

The images-per-equirect argument is the number of images that will be sampled from each equirectangular image. We have found that 8 images per equirectangular image is sufficient for most use cases so it defaults to that. However, if you find that there isn't enough detail in the nerf or that colmap is having trouble aligning the images, you can try increasing the number of images per equirectangular image to 14. See the video section below for details on cropping.

### Videos

For videos we recommend taking a video with the camera held on top of your head. This will result in any unwanted capturer to just be in the bottom of each frame image and therefore can be cropped out.

For a video, process the data using the following command:

```bash
ns-process-data video --camera-type equirectangular --images-per-equirect {8, or 14} --num-frames-target {num equirectangular frames to sample from} --crop-factor {top bottom left right} --data {data directory} --output-dir {output directory}
```

See the equirectangular images section above for a description of the `--images-per-equirect` argument.

The `num-frames-target` argument is optional but it is recommended to set it to 3*(seconds of video) frames. For example, if you have a 30 second video, you would use `--num-frames-target 90` (3*30=90). This number was chosen from a bit of experimentation and seems to work well for most videos. It is by no means a hard rule and you can experiment with different values.

The `crop-factor` argument is optional but often very helpful. This is because equirectangular videos taken by 360 cameras tend to have a portion of the bottom of the image that is the person who was holding the camera over their head.

  <img src="imgs/equirect_crop.jpg">

The pixels representing the distorted hand and head are obviously not useful in training a nerf so we can remove it by cropping the bottom 20% of the image. This can be done by using the `--crop-factor 0 0.2 0 0` argument.

If cropping only needs to be done from the bottom, you can use the `--crop-bottom [num]` argument which would be the same as doing `--crop-factor 0.0 [num] 0.0 0.0`

## 🥽 Render VR Video

Stereo equirectangular rendering for VR video is supported as VR180 and omni-directional stereo (360 VR) Nerfstudio camera types for video and image rendering. 

### Omni-directional Stereo (360 VR)
This outputs two equirectangular renders vertically stacked, one for each eye. Omni-directional stereo (ODS) is a method to render VR 3D 360 videos, and may introduce slight depth distortions for close objects. For additional information on how ODS works, refer to this [writeup](https://developers.google.com/vr/jump/rendering-ods-content.pdf).

<center>
<img img width="300" src="https://github-production-user-asset-6210df.s3.amazonaws.com/9502341/255423390-ff0710f1-29ce-47b2-85f9-922084cab297.jpg">
</center>


### VR180
This outputs two 180 deg equirectangular renders horizontally stacked, one for each eye. VR180 is a video format for VR 3D 180 videos. Unlike in omnidirectional stereo, VR180 content only displays front facing content. 

<center>
<img img width="375" src="https://github-production-user-asset-6210df.s3.amazonaws.com/9502341/255379444-b90f5b3c-5021-4659-8732-17725669914e.jpeg">
</center>

### Setup instructions
To render for VR video it is essential to adjust the NeRF to have an approximately true-to-life real world scale (adjustable in the camera path) to ensure that the scene depth and IPD (distance between the eyes) is appropriate for the render to be viewable in VR. You can adjust the scene scale with the [Nerfstudio Blender Add-on](https://docs.nerf.studio/extensions/blender_addon.html) by appropriately scaling a point cloud representation of the NeRF.
Results may be unviewable if the scale is not set appropriately. The IPD is set at 64mm by default but only is accurate when the NeRF scene is true to scale.

For good quality renders, it is recommended to render at high resolutions (For ODS: 4096x2048 per eye, or 2048x1024 per eye. For VR180: 4096x4096 per eye or 2048x2048 per eye). Render resolutions for a single eye are specified in the camera path. For VR180, resolutions must be in a 1:1 aspect ratio. For ODS, resolutions must be in a 2:1 aspect ratio. The final stacked render output will automatically be constructed (with aspect ratios for VR180 as 2:1 and ODS as 1:1).

:::{admonition} Note
:class: info
If you are rendering an image sequence, it is recommended to render as png instead of jpeg, since the png will appear clearer. However, file sizes can be significantly larger with png.
:::

To render with the VR videos camera:
1. Use the [Nerfstudio Blender Add-on](https://docs.nerf.studio/extensions/blender_addon.html) to set the scale of the NeRF scene and create the camera path
    - Export a point cloud representation of the NeRF
   - Import the point cloud representation in Blender and enable the Nerfstudio Blender Add-on
    - Create a reference object such as a cube which may be 1x1x1 meter. You could also create a cylinder and scale it to an appropriate height of a viewer.
    - Now scale the point cloud representation accordingly to match the reference object. This is to ensure that the NeRF scene is scaled as close to real life.
    - To place the camera at the correct height from the ground in the scene, you can create a cylinder representing the viewer vertically scaled to the viewer’s height, and place the camera at eye level.
    - Animate the camera movement as needed
    - Create the camera path JSON file with the Nerfstudio Blender Add-on

2. Edit the JSON camera path file

    **Omni-directional Stereo (360 VR)**
      - Open the camera path JSON file and specify the `camera_type` as `omnidirectional`
      - Specify the `render_height` and `render_width` to the resolution of a single eye. The width:height aspect ratio must be 2:1. Recommended resolutions are 4096x2048 or 2048x1024.
        <center>
        <img img width="250" src="https://github-production-user-asset-6210df.s3.amazonaws.com/9502341/240530527-22d14276-ac2c-46a5-a4b0-4785b7413241.png">
        </center>


    **VR180**
      - Open the camera path JSON file and specify the `camera_type` as `vr180`
      - Specify the `render_height` and `render_width` to the resolution of a single eye. The width:height aspect ratio must be 1:1. Recommended resolutions are 4096x4096 or 2048x2048.
      <center>
      <img img width="190" src="https://github-production-user-asset-6210df.s3.amazonaws.com/9502341/255379889-83b7fd09-ce8f-4868-8838-7be9b63f01b4.png">
      </center>

    - Save the camera path and render the NeRF


:::{admonition} Note
:class: info
If the depth of the scene is unviewable and looks too close or expanded when viewing the render in VR, the scale of the NeRF may be set too small. If there is almost no discernible depth, the scale of the NeRF may be too large. Getting the right scale may take some experimentation, so it is recommended to either render at a much lower resolution or just one frame to ensure the depth and render is viewable in the VR headset.
:::

#### Additional Notes
- Rendering with VR180 or ODS can take significantly longer than traditional renders due to higher resolutions and needing to render a left and right eye view for each frame. Render times may grow exponentially with larger resolutions.
- When rendering VR180 or ODS content, Nerfstudio will first render the left eye, then the right eye, and finally vertically stack the renders. During this process, Nerfstudio will create a temporary folder to store the left and right eye renders and delete this folder once the final renders are stacked.
- If rendering content where the camera is stationary for many frames, it is recommended to only render once at that position and extend the time in a video editor since ODS renders can take a lot of time to render.
- It is recommended to render a preliminary render at a much lower resolution or frame rate to test and ensure that the depth and camera position look accurate in VR.
 - The IPD can be modified in the `cameras.py` script as the variable `vr_ipd` (default is 64 mm).
 - Compositing with Blender Objects and VR180 or ODS Renders
   - Configure the Blender camera as panoramic and equirectangular. For the VR180 Blender camera, set the panoramic longitude min and max to -90 and 90.
   - Change the Stereoscopy mode to "Parallel" set the Interocular Distance to 0.064 m. 
## quickstart

### installation.md

# Installation

## Prerequisites

::::::{tab-set}
:::::{tab-item} Linux

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

:::::
:::::{tab-item} Windows

Install [Git](https://git-scm.com/downloads).

Install Visual Studio 2022. This must be done before installing CUDA. The necessary components are included in the `Desktop Development with C++` workflow (also called `C++ Build Tools` in the BuildTools edition).

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

:::::
::::::

## Create environment

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip

```

## Dependencies

(pytorch)=

### PyTorch

Note that if a PyTorch version prior to 2.0.1 is installed,
the previous version of pytorch, functorch, and tiny-cuda-nn should be uninstalled.

```bash
pip uninstall torch torchvision functorch tinycudann
```

::::{tab-set}
:::{tab-item} Torch 2.1.2 with CUDA 11.8 (recommended)

Install PyTorch 2.1.2 with CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

To build the necessary CUDA extensions, `cuda-toolkit` is also required. We
recommend installing with conda:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

:::
:::{tab-item} Torch 2.0.1 with CUDA 11.7

Install PyTorch 2.0.1 with CUDA 11.7:

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

To build the necessary CUDA extensions, `cuda-toolkit` is also required. We
recommend installing with conda:

```bash
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```

:::
::::

### tiny-cuda-nn

After pytorch and ninja, install the torch bindings for tiny-cuda-nn:

```bash
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## Installing nerfstudio

**From pip**

```bash
pip install nerfstudio
```

**From source**
Optional, use this command if you want the latest development version.

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

:::{admonition} Note
:class: info
Below are optional installations, but makes developing with nerfstudio much more convenient.
:::

**Tab completion (bash & zsh)**

This needs to be rerun when the CLI changes, for example if nerfstudio is updated.

```bash
ns-install-cli
```

**Development packages**

```bash
pip install -e .[dev]
pip install -e .[docs]
```

## Use docker image

Instead of installing and compiling prerequisites, setting up the environment and installing dependencies, a ready to use docker image is provided.

### Prerequisites

Docker ([get docker](https://docs.docker.com/get-docker/)) and nvidia GPU drivers ([get nvidia drivers](https://www.nvidia.de/Download/index.aspx?lang=de)), capable of working with CUDA 11.8, must be installed.
The docker image can then either be pulled from [here](https://hub.docker.com/r/dromni/nerfstudio/tags) (replace <version> with the actual version, e.g. 0.1.18)

```bash
docker pull dromni/nerfstudio:<version>
```

or be built from the repository using

```bash
docker build --tag nerfstudio -f Dockerfile .
```

To restrict to only CUDA architectures that you have available locally, use the `CUDA_ARCHITECTURES`
build arg and look up [the compute capability for your GPU](https://developer.nvidia.com/cuda-gpus).
For example, here's how to build with support for GeForce 30xx series GPUs:

```bash
docker build \
    --build-arg CUDA_VERSION=11.8.0 \
    --build-arg CUDA_ARCHITECTURES=86 \
    --build-arg OS_VERSION=22.04 \
    --tag nerfstudio-86 \
    --file Dockerfile .
```

The user inside the container is called 'user' and is mapped to the local user with ID 1000 (usually the first non-root user on Linux systems).  
If you suspect that your user might have a different id, override `USER_ID` during the build as follows:

```bash
docker build \
    --build-arg USER_ID=$(id -u) \
    --file Dockerfile .
```

### Using an interactive container

The docker container can be launched with an interactive terminal where nerfstudio commands can be entered as usual. Some parameters are required and some are strongly recommended for usage as following:

```bash
docker run --gpus all \                                         # Give the container access to nvidia GPU (required).
            -u $(id -u) \                                       # To prevent abusing of root privilege, please use custom user privilege to start.
            -v /folder/of/your/data:/workspace/ \               # Mount a folder from the local machine into the container to be able to process them (required).
            -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ \   # Mount cache folder to avoid re-downloading of models everytime (recommended).
            -p 7007:7007 \                                      # Map port from local machine to docker container (required to access the web interface/UI).
            --rm \                                              # Remove container after it is closed (recommended).
            -it \                                               # Start container in interactive mode.
            --shm-size=12gb \                                   # Increase memory assigned to container to avoid memory limitations, default is 64 MB (recommended).
            dromni/nerfstudio:<tag>                             # Docker image name if you pulled from docker hub.
            <--- OR --->
            nerfstudio                                          # Docker image tag if you built the image from the Dockerfile by yourself using the command from above.
```

### Call nerfstudio commands directly

Besides, the container can also directly be used by adding the nerfstudio command to the end.

```bash
docker run --gpus all -u $(id -u) -v /folder/of/your/data:/workspace/ -v /home/<YOUR_USER>/.cache/:/home/user/.cache/ -p 7007:7007 --rm -it --shm-size=12gb  # Parameters.
            dromni/nerfstudio:<tag> \                           # Docker image name
            ns-process-data video --data /workspace/video.mp4   # Smaple command of nerfstudio.
```

### Note

- The container works on Linux and Windows, depending on your OS some additional setup steps might be required to provide access to your GPU inside containers.
- Paths on Windows use backslash '\\' while unix based systems use a frontslash '/' for paths, where backslashes might require an escape character depending on where they are used (e.g. C:\\\\folder1\\\\folder2...). Alternatively, mounts can be quoted (e.g. `-v 'C:\local_folder:/docker_folder'`). Ensure to use the correct paths when mounting folders or providing paths as parameters.
- Always use full paths, relative paths are known to create issues when being used in mounts into docker.
- Everything inside the container, what is not in a mounted folder (workspace in the above example), will be permanently removed after destroying the container. Always do all your tasks and output folder in workdir!
- The container currently is based on nvidia/cuda:11.8.0-devel-ubuntu22.04, consequently it comes with CUDA 11.8 which must be supported by the nvidia driver. No local CUDA installation is required or will be affected by using the docker image.
- The docker image (respectively Ubuntu 22.04) comes with Python3.10, no older version of Python is installed.
- If you call the container with commands directly, you still might want to add the interactive terminal ('-it') flag to get live log outputs of the nerfstudio scripts. In case the container is used in an automated environment the flag should be discarded.
- The current version of docker is built for multi-architecture (CUDA architectures) use. The target architecture(s) must be defined at build time for Colmap and tinyCUDNN to be able to compile properly. If your GPU architecture is not covered by the following table you need to replace the number in the line `ARG CUDA_ARCHITECTURES=90;89;86;80;75;70;61;52;37` to your specific architecture. It also is a good idea to remove all architectures but yours (e.g. `ARG CUDA_ARCHITECTURES=86`) to speedup the docker build process a lot.
- To avoid memory issues or limitations during processing, it is recommended to use either `--shm-size=12gb` or `--ipc=host` to increase the memory available to the docker container. 12gb as in the example is only a suggestion and may be replaced by other values depending on your hardware and requirements.

**Currently supported CUDA architectures in the docker image**

(tiny-cuda-arch-list)=

| GPU             | CUDA arch |
| --------------- | --------- |
| H100            | 90        |
| 40X0            | 89        |
| 30X0            | 86        |
| A100            | 80        |
| 20X0            | 75        |
| TITAN V / V100  | 70        |
| 10X0 / TITAN Xp | 61        |
| 9X0             | 52        |
| K80             | 37        |

## Installation FAQ

- [ImportError: DLL load failed while importing \_89_C](tiny-cuda-mismatch-arch)
- [tiny-cuda-nn installation errors out with cuda mismatch](tiny-cuda-mismatch-error)
- [tiny-cuda-nn installation errors out with no CUDA toolset found](tiny-cuda-integration-error)
- [Installation errors, File "setup.py" not found](pip-install-error)
- [Runtime errors, "len(sources) > 0".](cuda-sources-error)

 <br />

(tiny-cuda-mismatch-arch)=

**ImportError: DLL load failed while importing \_89_C**

This occurs with certain GPUs that have CUDA architecture versions (89 in the example above) for which tiny-cuda-nn does not automatically compile support.

**Solution**:

Reinstall tiny-cuda-nn with the following command:

::::::{tab-set}
:::::{tab-item} Linux

```bash
TCNN_CUDA_ARCHITECTURES=XX pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
:::::{tab-item} Windows

```bash
set TCNN_CUDA_ARCHITECTURES=XX
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

:::::
::::::

Where XX is the architecture version listed [here](tiny-cuda-arch-list). Ie. for a 4090 GPU use `TCNN_CUDA_ARCHITECTURES=89`

 <br />

(tiny-cuda-mismatch-error)=

**tiny-cuda-nn installation errors out with cuda mismatch**

While installing tiny-cuda, you run into: `The detected CUDA version mismatches the version that was used to compile PyTorch (10.2). Please make sure to use the same CUDA versions.`

**Solution**:

Reinstall PyTorch with the correct CUDA version.
See [pytorch](pytorch) under Dependencies, above.

 <br />

(tiny-cuda-integration-error)=

**(Windows) tiny-cuda-nn installation errors out with no CUDA toolset found**

While installing tiny-cuda on Windows, you run into: `No CUDA toolset found.`

**Solution**:

Confirm that you have Visual Studio installed.

Make sure CUDA Visual Studio integration is enabled. This should be done automatically by the CUDA installer if it is run after Visual Studio is installed. You can also manually enable integration.

::::{tab-set}
:::{tab-item} Visual Studio 2019

To manually enable integration for Visual Studio 2019, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
:::{tab-item} Visual Studio 2022

To manually enable integration for Visual Studio 2022, copy all 4 files from

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files\Microsoft Visual Studio\2022\[Community, Professional, Enterprise, or BuildTools]\MSBuild\Microsoft\VC\v160\BuildCustomizations
```

:::
::::

 <br />

(pip-install-error)=

**Installation errors, File "setup.py" not found**

When installing dependencies and nerfstudio with `pip install -e .`, you run into: `ERROR: File "setup.py" not found. Directory cannot be installed in editable mode`

**Solution**:
This can be fixed by upgrading pip to the latest version:

```
python -m pip install --upgrade pip
```

 <br />

(cuda-sources-error)=

**Runtime errors: "len(sources) > 0", "ctype = \_C.ContractionType(type.value) ; TypeError: 'NoneType' object is not callable".**

When running `train.py `, an error occurs when installing cuda files in the backend code.

**Solution**:
This is a problem with not being able to detect the correct CUDA version, and can be fixed by updating the CUDA path environment variables:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export PATH=$PATH:$CUDA_HOME/bin
```
## quickstart

### viewer_quickstart.rst

```rst
Using the viewer
================

The nerfstudio web-based viewer makes it easy to view training in real-time, and to create content videos from your trained models 🌟!

Connecting to the viewer
^^^^^^^^^^^^^^^^^^^^^^^^

The nerfstudio viewer is launched automatically with each :code:`ns-train` training run! It can also be run separately with :code:`ns-viewer`.
To access a viewer, we need to enter the server's address and port in a web browser.

Accessing on a local machine
""""""""""""""""""""""""""""

You should be able to click the link obtained while running a script to open the viewer in your browser. This should typically look something like :code:`http://localhost:7007`.

Accessing over an SSH connection
""""""""""""""""""""""""""""""""

If you are training on a remote machine, the viewer will still let you view your trainings. You will need to forward traffic from your viewing host to the listening port on the host running ns-train (7007 by default). You can achieve this by securely tunneling traffic on the specified port through an ssh session. In a terminal window on the viewing host, issue the following command:

..  code-block:: bash

   ssh -L 7007:127.0.0.1:7007 <username>@<training-host-ip>


..  admonition:: Note

    You can now simply open the link (same one shown in image above) in your browser on your local machine and it should connect!. So long as you don't close this terminal window with this specific active ssh connection, the port will remain open for all your other ssh sessions.

    For example, if you do this in a new terminal window, any existing ssh sessions (terminal windows, VSCode remote connection, etc) will still be able to access the port, but if you close this terminal window, the port will be closed.


..  warning::
    If the port is being used, you will need to switch the port using the `--viewer.websocket-port` flag tied to the model subcommand.


Accessing via a share link
""""""""""""""""""""""""""

To connect to remote machines without requiring an SSH connection, we also support creating share URLs.
This can be generated from the "Share:" icon in the GUI, or from the CLI by specifying the :code:`--viewer.make-share-url True` argument.
This will create a publically accessible link that you can share with others to view your training progress.
This is useful for sharing training progress with collaborators, or for viewing training progress on a remote machine without requiring an SSH connection.
However, it will introduce some latency in the viewer.


..  seealso::

  For a more in-depth developer overview on how to hack with the viewer, see our `developer guide </docs/_build/html/developer_guides/viewer/index.html>`_


Legacy viewer tutorial video
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the tutorial video below, we walk you through how you can turn a simple capture into a 3D video 📸 🎬

This video is for the legacy viewer, which was the default in :code:`nerfstudio<1.0`. It has some visual differences from the current viewer. However, the core functionality is the same.
The legacy viewer can still be enabled with the :code:`--vis viewer_legacy` option, for example, :code:`ns-train nerfacto --vis viewer_legacy`.

.. raw:: html

   <iframe width="560" height="315" src="https://www.youtube.com/embed/nSFsugarWzk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

For specific sections, click the links below to navigate to the associated portion of the video.

Getting started
"""""""""""""""

* `Hello from nerfstudio <https://youtu.be/nSFsugarWzk?t=0>`_
* `Preprocess your video <https://youtu.be/nSFsugarWzk?t=13>`_
* `Launching training and viewer <https://youtu.be/nSFsugarWzk?t=27>`_

Viewer basics
"""""""""""""""

* `Viewer scene introduction <https://youtu.be/nSFsugarWzk?t=63>`_
* `Moving around in viewer <https://youtu.be/nSFsugarWzk?t=80>`_
* `Overview of Controls Panel - train speed/output options <https://youtu.be/nSFsugarWzk?t=98>`_
* `Overview of Scene Panel - toggle visibility <https://youtu.be/nSFsugarWzk?t=115>`_

Creating camera trajectories
""""""""""""""""""""""""""""

* `Creating a custom camera path <https://youtu.be/nSFsugarWzk?t=136>`_
* `Camera spline options - cycle, speed, smoothness <https://youtu.be/nSFsugarWzk?t=158>`_
* `Camera options - move, add, view <https://youtu.be/nSFsugarWzk?t=177>`_

Rendering a video
"""""""""""""""""

* `Preview camera trajectory <https://youtu.be/nSFsugarWzk?t=206>`_
* `How to render final video <https://youtu.be/nSFsugarWzk?t=227>`_

|

```

## quickstart

### export_geometry.md

# Export geometry

Here we document how to export point clouds and meshes from nerfstudio. The main command you'll be working with is `ns-export`. Our point clouds are exported as `.ply` files and the textured meshes are exported as `.obj` files.

## Exporting a mesh

### 1. TSDF Fusion

TSDF (truncated signed distance function) Fusion is a meshing algorithm that uses depth maps to extract a surface as a mesh. This method works for all models.

```python
ns-export tsdf --load-config CONFIG.yml --output-dir OUTPUT_DIR
```

### 2. Poisson surface reconstruction

Poisson surface reconstruction gives the highest quality meshes. See the steps below to use Poisson surface reconstruction in our repo.

> **Note:**
> This will only work with a Model that computes or predicts normals, e.g., nerfacto.

1. Train nerfacto with network settings that predict normals.

```bash
ns-train nerfacto --pipeline.model.predict-normals True
```

2. Export a mesh with the Poisson meshing algorithm.

```bash
ns-export poisson --load-config CONFIG.yml --output-dir OUTPUT_DIR
```

## Exporting a point cloud

```bash
ns-export pointcloud --help
```

## Other exporting methods

Run the following command to see other export methods that may exist.

```python
ns-export --help
```

## Texturing an existing mesh with NeRF

Say you want to simplify and/or smooth a mesh offline, and then you want to texture it with NeRF. You can do that with the following command. It will work for any mesh filetypes that [PyMeshLab](https://pymeshlab.readthedocs.io/en/latest/) can support, for example a `.ply`.

```python
python nerfstudio/scripts/texture.py --load-config CONFIG.yml --input-mesh-filename FILENAME --output-dir OUTPUT_DIR
```

## Dependencies

Our dependencies are shipped with the pip package in the pyproject.toml file. These are the following:

- [xatlas-python](https://github.com/mworchel/xatlas-python) for unwrapping meshes to a UV map
- [pymeshlab](https://pymeshlab.readthedocs.io/en/latest/) for reducing the number of faces in a mesh
## extensions

### unreal_engine.md

# Exporting to Unreal Engine

 ```{image} imgs/desolation_unreal.png
 :width: 800 
 :align: center 
 :alt: NeRF in Unreal Engine 
 ``` 

## Overview

NeRFStudio models can be used in Unreal Engine if they are converted to an NVOL file. NVOL is a new standard file format to store NeRFs in a fast and efficient way. NVOL files can be obtained from NeRFStudio checkpoints files (.ckpt) using the [Volinga Suite](https://volinga.ai/).


## Exporting your model to NVOL
Currently NVOL file only supports Volinga model (which is based on nerfacto). To use Volinga model you will need to install [Volinga extension for NeRFStudio](https://github.com/Volinga/volinga-model). You can train your model using the following command:

```bash
ns-train volinga --data /path/to/your/data --vis viewer
```

Once the training is done, you can find your checkpoint file in the `outputs/path-to-your-data/volinga` folder. Then, you can drag it to Volinga Suite to export it to NVOL.

 ```{image} imgs/export_nvol.png 
 :width: 400 
 :align: center 
 :alt: Nvol export in Voliga Suite 
 ``` 

Once the NVOL is ready, you can download it and use it in Unreal Engine.

 ```{image} imgs/nvol_ready.png 
 :width: 800 
 :align: center 
 :alt: NVOL ready to use 
 ``` ## extensions

### blender_addon.md

# Blender VFX add-on

<p align="center">
    <iframe width="728" height="409" src="https://www.youtube.com/embed/vDhj6j7kfWM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</p>

## Overview

This Blender add-on allows for compositing with a Nerfstudio render as a background layer by generating a camera path JSON file from the Blender camera path, as well as a way to import Nerfstudio JSON files as a Blender camera baked with the Nerfstudio camera path. This add-on also allows compositing multiple NeRF objects into a NeRF scene. This is achieved by importing a mesh or point-cloud representation of the NeRF scene from Nerfstudio into Blender and getting the camera coordinates relative to the transformations of the NeRF representation. Dynamic FOV from the Blender camera is supported and will match the Nerfstudio render. Perspective, equirectangular, VR180, and omnidirectional stereo (VR 360) cameras are supported. This add-on also supports Gaussian Splatting scenes as well, however equirectangular and VR video rendering is not currently supported for splats.

<center>
 <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211442247-99d1ebc7-3ef9-46f7-9bcc-0e18553f19b7.PNG">
</center>

## Add-on Setup Instructions

1. The add-on requires Blender 3.0 or newer, install Blender [here](https://www.blender.org/).

2. Download <a href="https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/nerfstudio/scripts/blender/nerfstudio_blender.py" download="nerfstudio_blender.py">Blender Add-on Script</a>

3. Install and enable the add-on in Blender in `Edit → Preferences → Add-Ons`. The add-on will be visible in the Render Properties tab on the right panel.
    <center>
   <img width="500" alt="image" src="https://user-images.githubusercontent.com/9502341/232202430-d4a38ac7-2566-4975-97a4-76220f336511.png">
   </center>

4. The add-on should now be installed in the `Render Properties` panel
    <center>
   <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/232202091-c13c66c4-f119-4f15-aa3e-3bf736371821.png">
   </center>

## Scene Setup

1. Export the mesh or point cloud representation of the NeRF from Nerfstudio, which will be used as reference for the actual NeRF in the Blender scene. Mesh export at a good quality is preferred, however, if the export is not clear or the NeRF is large, a detailed point cloud export will also work. Keep the `save_world_frame` flag as False or in the viewer, de-select the "Save in world frame" checkbox to keep the correct coordinate system for the add-on.

2. Import the mesh or point cloud representation of the NeRF into the scene. You may need to crop the mesh further. Since it is used as a reference and won't be visible in the final render, only the parts that the blender animation will interact with may be necessary to import.

3. Select the NeRF mesh or point cloud in the add-on.

4. Resize, position, or rotate the NeRF representation to fit your scene.

## Generate Nerfstudio JSON Camera Path from Blender Camera

1. There are a few ways to hide the reference mesh for the Blender render

   - In object properties, select "Shadow Catcher". This makes the representation invisible in the render, but all shadows cast on it will render. You may have to switch to the cycles renderer to see the shadow catcher option.
   <center>
   <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/211244787-859ca9b5-6ba2-4056-aaf2-c89fc6370c2a.png">
   </center>

   - Note: This may not give ideal results if the mesh is not very clear or occludes other objects in the scene. If this is the case, you can hide the mesh from the render instead by clicking the camera button in the Outliner next to its name.
   <center>
   <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/211244858-54091d36-086d-4211-a7d9-75dcfdcb1436.png">
   </center>

2. Verify that the animation plays and the NeRF representation does not occlude the camera.
3. Go to the Nerfstudio Add-on in Render Properties and expand the "Nerfstudio Path Generator" tab in the panel. Use the object selector to select the NeRF representation. Then, select the file path for the output JSON camera path file.
4. Click "Generate JSON file". The output JSON file is named `camera_path_blender.json`.
<center>
<img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211442361-999fe040-a1ed-43f0-b079-0c659d70862f.png">
</center>

5. Render the NeRF with the generated camera path using Nerfstudio in the command prompt or terminal.

6. Before rendering the Blender animation, go to the Render Properties and in the Film settings select "Transparent" so that the render will be rendered with a clear background to allow it to be composited over the NeRF render.
<center>
<img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211244801-c555c3b5-ab3f-4d84-9f64-68559b03ff37.png">
</center>

7. Now the scene can be rendered and composited over the camera aligned Nerfstudio render.
<center>
<img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211245025-2ef5adbe-9306-4eab-b761-c78e3ec187bd.png">
</center>

### Examples

<p align="center">
    <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/212463397-ba34d60f-a744-47a1-95ce-da6945a7fc00.gif">
    <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/212461881-e2096710-4732-4f20-9ba7-1aeb0d0d653b.gif">
</p>

### Additional details

- You can also apply an environment texture to the Blender scene by using Nerfstudio to render an equirectangular image (360 image) of the NeRF from a place in the scene.
- The settings for the Blender equirectangular camera are: "Panoramic" camera type and panorama type of "Equirectangular".
    <center>
    <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/211245895-76dfb65d-ed81-4c36-984a-4c683fc0e1b4.png">
    </center>

- The generated JSON camera path follows the user specified frame start, end, and step fields in the Output Properties in Blender. The JSON file specifies the user specified x and y render resolutions at the given % in the Output Properties.
- The add-on computes the camera coordinates based on the active camera in the scene.
- FOV animated changes of the camera will be matched with the NeRF render.
- Perspective, equirectangular, VR180, and omnidirectional stereo cameras are supported and can be configured within Blender.
- The generated JSON file can be imported into Nerfstudio. Each keyframe of the camera transform in the frame sequence in Blender will be a keyframe in Nerfstudio. The exported JSON camera path is baked, where each frame in the sequence is a keyframe. This is to ensure that frame interpolation across Nerfstudio and Blender do not differ.
    <center>
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211245108-587a97f7-d48d-4515-b09a-5644fd966298.png">
    </center>
- It is recommended to run the camera path in the Nerfstudio web interface with the NeRF to ensure that the NeRF is visible throughout the camera path.
- The fps and keyframe timestamps are based on the "Frame Rate" setting in the Output Properties in Blender.
- The NeRF representation can also be transformed (position, rotation, and scale) as well as animated.
- The pivot point of the NeRF representation should not be changed
- It is recommended to export a high fidelity mesh as the NeRF representation from Nerfstudio. However if that is not possible or the scene is too large, a point cloud representation also works.
- For compositing, it is recommended to convert the video render into image frames to ensure the Blender and NeRF renders are in sync.
- Currently, dynamic camera focus is not supported.
- Compositing with Blender Objects and VR180 or ODS Renders
  - Configure the Blender camera to be panoramic equirectangular and enable stereoscopy in the Output Properties. For the VR180 Blender camera, set the panoramic longitude min and max to -90 and 90.
  - Under the Stereoscopy panel the Blender camera settings, change the mode to "Parallel", set the Interocular Distance to 0.064 m, and checkmark "Spherical Stereo".

    <center>
    <img width="300" alt="image" src="https://github-production-user-asset-6210df.s3.amazonaws.com/9502341/253217833-fd607601-2b81-48ab-ac5d-e55514a588da.png">
    </center>
- Fisheye and orthographic cameras are not supported.
- Renders with Gaussian Splats are supported, but the point cloud or mesh representation would need to be generated from training a NeRF on the same dataset.
- A walkthrough of this section is included in the tutorial video.

## Create Blender Camera from Nerfstudio JSON Camera Path

<p align="center">
    <img width="800" alt="image" src="https://user-images.githubusercontent.com/9502341/211246016-88bc6031-01d4-418f-8230-fb8c29212200.png">
</p>

1. Expand the "Nerfstudio Camera Generator" tab in the panel. After inputting the NeRF representation, select the JSON Nerfstudio file and click "Create Camera from JSON"

2. A new camera named "NerfstudioCamera" should appear in the viewport with the camera path and FOV of the input file. This camera's type will match the Nerfstudio input file, except fisheye cameras.

### Additional details

- Since only the camera path, camera type, and FOV are applied on the created Blender camera, the timing and duration of the animation may need to be adjusted.
- Fisheye cameras imported from Nerfstudio are not supported and will default to perspective cameras.
- Animated NeRF representations will not be reflected in the imported camera path animation.
- The newly created camera is not active by default, so you may need to right click and select "Set Active Camera".
- The newly created Blender camera animation is baked, where each frame in the sequence is a keyframe. This is to ensure that frame interpolation across Nerfstudio and Blender do not differ.
- The resolution settings from the input JSON file do not affect the Blender render settings
- Scale of the camera is not keyframed from the Nerfstudio camera path.
- This newly created camera has a sensor fit of "Vertical"

## Compositing NeRF Objects in NeRF Environments

You can composite NeRF objects into a scene with a NeRF background by rendering the cropped NeRF object along with an accumulation render as an alpha mask and compositing that over the background NeRF render.

<p align="center">
    <img width="450" alt="image" src="https://user-images.githubusercontent.com/9502341/232261745-80e36ae1-8527-4256-bbd0-83461e6f4324.jpg">
</p>

1. Import the background NeRF scene as a point cloud (or mesh, but point cloud is preferred for large scenes).

2. Export a cropped NeRF mesh of the NeRF object(s)

   - Open the NeRF object from the Nerfstudio viewer and select "Crop Viewport" and accordingly adjust the Scale and Center values of the bounding box to crop the NeRF scene to around the bounds of the object of interest.

   <center>
   <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/232264554-26357747-6f09-4084-9710-dabf60c61909.jpg">
   </center>

   - Copy over the scale and center values into the Export panel and export the NeRF object as a mesh (point cloud will also work but shadows can be rendered if the object is a mesh)

   - Keep note of the scale and center values for the crop. This can be done by creating a new JSON camera path in the editor which will add a crop section towards the end of the file.

   <center>
   <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/232264430-0b991e20-838f-4e06-9834-7d1d8c1d0042.png">
   </center>

3. Import the NeRF object representation. Rescale and position the scene and NeRF object and background environment. You can also animate either of them.

4. (Optional) To add shadows of the NeRF object(s)

   - Add a plane representing the ground of the environment. In object properties, select "Shadow Catcher" under Visibility. You may have to switch to the cycles renderer to see the shadow catcher option.

   - In the object properties of the NeRF object, go to the Ray Visibility section and deselect the "Camera" option. This will hide the mesh in the Blender render, but keep its shadow.

   <center>
   <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/232265212-ca077af8-fb3a-491f-8d60-304cb9fae57e.png">
   </center>

   - Render the Blender animation with only the shadow of the object on the shadow catcher visible. Render with a transparent background by selecting "Transparent" under the Film Settings in the Render Properties.

5. Go to the Nerfstudio Add-on in Render Properties and expand the "Nerfstudio Path Generator" tab in the panel. Use the object selector to select the NeRF environment representation. Then, select the file path for the output JSON camera path file. Click "Generate JSON file". The output JSON file is named `camera_path_blender.json`. It is recommended to rename this JSON file to keep track of the multiple camera paths that will be used to construct the full render.

6. Generate a Nerfstudio camera path now for the NeRF object. Select the object selector to select the NeRF object. Before generating the new JSON file, you may need to rename the previously generated camera path for the environment or move it to a different directory otherwise the new JSON file will overwrite it. Click "Generate JSON file". The new output JSON file is named `camera_path_blender.json`. You will need to repeat this process for each NeRF object you want to composite in your scene.

7. Render the NeRF of the background environment using the generated camera path for the environment using the Nerfstudio in the command line or terminal.

8. Render the NeRF object

   - Open the recently generated blender Nerfstudio JSON camera path and add the crop parameters to the camera path. This section will be placed towards the end of the Blender Nerfstudio camera path after the `is_cycle` field. This can be done by copying over the crop section from the JSON camera path with the scene crop which was exported earlier. Alternatively, you can enter the crop values manually in this format.
   <center>
   <img width="200" alt="image" src="https://user-images.githubusercontent.com/9502341/232264689-d2e7747c-ce53-425b-810b-1b2954eceb62.png">
   </center>

   - Render the NeRF object using Nerfstudio and the edited camera path in the command line or terminal. This will be the RGB render.

   - Next, render the accumulation render as an alpha mask of the NeRF object by adding the command line argument `--rendered-output-names accumulation` to the render command.

   <center>
   <img width="450" alt="image" src="https://user-images.githubusercontent.com/9502341/232264813-195c970f-b194-4761-843b-983123f128f7.jpg">
   </center>

9. Convert each of the Nerfstudio render videos into an image sequence of frames as PNG or JPG files. This will ensure that the frames will be aligned when compositing. You can convert the video mp4 to an image sequence by creating a Blender Video Editing file and rendering the mp4 as JPGs or PNGs.

10. Composite the NeRF renders in a video editing software such as Adobe Premiere Pro.

    - Place the render of the Nerfstudio background environment at the lowest layer, then place the shadow render of the NeRF object if created.

    - Place the RGB NeRF render of the NeRF object over the environment (and shadow if present) layers and then place the accumulation NeRF object render over the RGB NeRF object render.

    - Apply a filter to use the accumulation render as an alpha mask. In Premiere Pro, apply the effect "Track Matte Key" to the RGB render and select the "Matte" as the video track of the accumulation render and under "Composite Using" select "Matte Luma".

### Additional Details

- The RGB and accumulation renders will need to be rendered for each NeRF cropped object in the scene, but not for the NeRF environment if it is not cropped.
- If you will composite a shadow layer, the quality of the exported mesh of the NeRF object should have enough fidelity to cast a shadow, but the texture doesn't need to be clear.
- If motion tracking or compositing over real camera footage, you can add planes or cubes to represent walls or doorways as shadow catcher or holdout objects. This will composite the shadow layer over the NeRF environment and help create alpha masks.
- The pivot point of the NeRF representations should not be changed.
- A walkthrough of this section is included in the tutorial video.

### Examples

<p align="center">
    <img width="300" alt="image" src="https://user-images.githubusercontent.com/9502341/232274012-27fb912c-3d3e-47b2-bb6b-68abf8f0692a.gif">
    <img width="225" alt="image" src="https://user-images.githubusercontent.com/9502341/232274049-d03e9768-8905-4668-b41b-d8ad4f122829.gif">
</p>

## Implementation Details

For generating the JSON camera path, we iterate over the scene frame sequence (from the start to the end with step intervals) and get the camera 4x4 world matrix at each frame. The world transformation matrix gives the position, rotation, and scale of the camera. We then obtain the world matrix of the NeRF representation at each frame and transform the camera coordinates with this to get the final camera world matrix. This allows us to re-position, rotate, and scale the NeRF representation in Blender and generate the right camera path to render the NeRF accordingly in Nerfstudio. Additionally, we calculate the FOV of the camera at each frame based on the sensor fit (horizontal or vertical), angle of view, and aspect ratio.
Next, we construct the list of keyframes which is very similar to the world matrices of the transformed camera matrix.
Camera properties in the JSON file are based on user specified fields such as resolution (user specified in Output Properties in Blender), camera type (Perspective or Equirectangular). In the JSON file, `aspect` is specified as 1.0, `smoothness_value` is set to 0, and `is_cycle` is set to false. The Nerfstudio render is the fps specified in Blender where the duration is the total number of frames divided by the fps.
Finally, we construct the full JSON object and write it to the file path specified by the user.

For generating the camera from the JSON file, we create a new Blender camera based on the input file and iterate through the `camera_path` field in the JSON to get the world matrix of the object from the `matrix_to_world` and similarly get the FOV from the `fov` fields. At each iteration, we set the camera to these parameters and insert a keyframe based on the position, rotation, and scale of the camera as well as the focal length of the camera based on the vertical FOV input.
## extensions

### sdfstudio.md

# SDFStudio

[project website](https://autonomousvision.github.io/sdfstudio/)

```{image} imgs/sdfstudio_overview.svg
:width: 800
:align: center
:alt: sdfstudio overview figure
```

## Overview

SDFStudio is built on top of nerfstudio. It implements multiple implicit surface reconstruction methods including:

- UniSurf
- VolSDF
- NeuS
- MonoSDF
- Mono-UniSurf
- Mono-NeuS
- Geo-NeuS
- Geo-UniSurf
- Geo-VolSDF
- NeuS-acc
- NeuS-facto
- NeuralReconW

You can learn more about these methods [here](https://github.com/autonomousvision/sdfstudio/blob/master/docs/sdfstudio-methods.md#Methods)

## Surface models in nerfstudio

We intend to integrate many of the SDFStudio improvements back into the nerfstudio core repository.

Supported methods:

- NeuS
- NeuS-facto

## Citation

If you use these surface based models in your research, you should consider citing the authors of SDFStudio,

```none
@misc{Yu2022SDFStudio,
    author    = {Yu, Zehao and Chen, Anpei and Antic, Bozidar and Peng, Songyou Peng and Bhattacharyya, Apratim
                 and Niemeyer, Michael and Tang, Siyu and Sattler, Torsten and Geiger, Andreas},
    title     = {SDFStudio: A Unified Framework for Surface Reconstruction},
    year      = {2022},
    url       = {https://github.com/autonomousvision/sdfstudio},
}
```
## nerfology/model_components

### index.md

# Model components

It can be difficult getting started with NeRFs. The research field is still quite new and most of the key nuggets are buried in academic papers. For this reason, we have consolidated many of the key concepts into a series of guides.

```{toctree}
    :maxdepth: 1
    Cameras models<visualize_cameras.ipynb>
    Sample representation<visualize_samples.ipynb>
    Ray samplers<visualize_samplers.ipynb>
    Spatial distortions<visualize_spatial_distortions.ipynb>
    Encoders<visualize_encoders.ipynb>
```
## nerfology/methods

### in2n.md

# Instruct-NeRF2NeRF

<h4>Editing 3D Scenes with Instructions</h4>

```{button-link} https://instruct-nerf2nerf.github.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/ayaanzhaque/instruct-nerf2nerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-nerf2nerf.github.io/data/videos/face.mp4" type="video/mp4">
</video>

**Instruct-NeRF2NeRF enables instruction-based editing of NeRFs via a 2D diffusion model**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/ayaanzhaque/instruct-nerf2nerf
```

## Running Instruct-NeRF2NeRF

Details for running Instruct-NeRF2NeRF (built with Nerfstudio!) can be found [here](https://github.com/ayaanzhaque/instruct-nerf2nerf). Once installed, run:

```bash
ns-train in2n --help
```

Three variants of Instruct-NeRF2NeRF are provided:

| Method       | Description                  | Memory | Quality |
| ------------ | ---------------------------- | ------ | ------- |
| `in2n`       | Full model, used in paper    | ~15GB  | Best    |
| `in2n-small` | Half precision model         | ~12GB  | Good    |
| `in2n-tiny`  | Half prevision with no LPIPS | ~10GB  | Ok      |

## Method

### Overview

Instruct-NeRF2NeRF is a method for editing NeRF scenes with text-instructions. Given a NeRF of a scene and the collection of images used to reconstruct it, the method uses an image-conditioned diffusion model ([InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix)) to iteratively edit the input images while optimizing the underlying scene, resulting in an optimized 3D scene that respects the edit instruction. The paper demonstrates that their method is able to edit large-scale, real-world scenes, and is able to accomplish more realistic, targeted edits than prior work.

## Pipeline

<video id="pipeline" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-nerf2nerf.github.io/data/videos/pipeline_animation.mp4" type="video/mp4">
</video>

This section will walk through each component of the Instruct-NeRF2NeRF method.

### How it Works

Instruct-NeRF2NeRF gradually updates a reconstructed NeRF scene by iteratively updating the dataset images while training the NeRF:

1. An image is rendered from the scene at a training viewpoint.
2. It is edited by InstructPix2Pix given a global text instruction.
3. The training dataset image is replaced with the edited image.
4. The NeRF continues training as usual.

### Editing Images with InstructPix2Pix

InstructPix2Pix is an image-editing diffusion model which can be prompted using text instructions. More details on InstructPix2Pix can be found [here](https://www.timothybrooks.com/instruct-pix2pix).

Traditionally, at inference time, InstructPix2Pix takes as input random noise and is conditioned on an image (the image to edit) and a text instruction. The strength of the edit can be controlled using the image and text classifier-free guidance scales.

To update a dataset image a given viewpoint, Instruct-NeRF2NeRF first takes the original, unedited training image as image conditioning and uses the global text instruction as text conditioning. The main input to the diffusion model is a noised version of the current render from the given viewpoint. The noise is sampled from a normal distribution and scaled based on a randomly chosen timestep. Then InstructPix2Pix slowly denoises the rendered image by predicting the noised version of the image at previous timesteps until the image is fully denoised. This will produce an edited version of the input image.

This process mixes the information of the diffusion model, which attempts to edit the image, the current 3D structure of the NeRF, and view-consistent information from the unedited, ground-truth images. By combining this set of information, the edit is respected while maintaining 3D consistency.

The code snippet for how an image is edited in the pipeline can be found [here](https://github.com/ayaanzhaque/instruct-nerf2nerf/blob/main/in2n/ip2p.py).

### Iterative Dataset Update

When NeRF training starts, the dataset consists of the original, unedited images used to train the original scene. These images are saved separately to use as conditioning for InstructPix2Pix. At each optimization step, some number of NeRF optimization steps are performed, and then some number of images (often just one) are updated. The images are randomly ordered prior to training and then at each step, the images are chosen in order to edit. Once an image has been edited, it is replaced in the dataset. Importantly, at each NeRF step, rays are sampled across the entire dataset, meaning there is a mixed source of supervision between edited images and unedited images. This allows for a gradual optimization that balances maintaining the 3D structure and consistency of the NeRF as well as performing the edit.

At early iterations of this process, the edited images may be inconsistent with one another, as InstructPix2Pix often doesn't perform consistent edits across viewpoints. However, over time, since images are edited using the current render of the NeRF, the edits begin to converge towards a globally consistent depiction of the underlying scene. Here is an example of how the underlying dataset evolves and becomes more consistent.

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-nerf2nerf.github.io/data/videos/du_update.mp4" type="video/mp4">
</video>

The traditional method for supervising NeRFs using diffusion models is to use a Score Distillation Sampling (SDS) loss, as proposed in [DreamFusion](https://dreamfusion3d.github.io/). The Iterative Dataset Update method can be viewed as a variant of SDS, as instead of updating a discrete set of images at each step, the loss is a mix of rays from various viewpoints which are edited to varying degrees. The results show that this leads to higher quality performance and more stable optimization.

## Results

For results, view the [project page](https://instruct-nerf2nerf.github.io/)!
## nerfology/methods

### splat.md

# Splatfacto
<h4>Nerfstudio's Gaussian Splatting Implementation</h4>
<iframe width="560" height="315" src="https://www.youtube.com/embed/0yueTFx-MdQ?si=GxiYnFAeYVVl-soJ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

```{button-link} https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
:color: primary
:outline:
Paper Website
```

[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) was proposed in SIGGRAPH 2023 from INRIA, and is a completely different method of representing radiance fields by explicitly storing a collection of 3D volumetric gaussians. These can be "splatted", or projected, onto a 2D image provided a camera pose, and rasterized to obtain per-pixel colors. Because rasterization is very fast on GPUs, this method can render much faster than neural representations of radiance fields.

To avoid confusion with the original paper, we refer to nerfstudio's implementation as "Splatfacto", which will drift away from the original as more features are added. Just as Nerfacto is a blend of various different methods, Splatfacto will be a blend of different gaussian splatting methodologies.

### Installation

```{button-link} https://docs.gsplat.studio/
:color: primary
:outline:
GSplat 
```

Nerfstudio uses [gsplat](https://github.com/nerfstudio-project/gsplat) as its gaussian rasterization backend, an in-house re-implementation which is designed to be more developer friendly. This can be installed with `pip install gsplat`. The associated CUDA code will be compiled the first time gsplat is executed. Some users with PyTorch 2.0 have experienced issues with this, which can be resolved by either installing gsplat from source, or upgrading torch to 2.1.

### Data
Gaussian splatting works much better if you initialize it from pre-existing geometry, such as SfM points from COLMAP. COLMAP datasets or datasets from `ns-process-data` will automatically save these points and initialize gaussians on them. Other datasets currently do not support initialization, and will initialize gaussians randomly. Initializing from other data inputs (i.e. depth from phone app scanners) may be supported in the future.

Because the method trains on *full images* instead of bundles of rays, there is a new datamanager in `full_images_datamanager.py` which undistorts input images, caches them, and provides single images at each train step.


### Running the Method
To run splatfacto, run `ns-train splatfacto --data <data>`. Just like NeRF methods, the splat can be interactively viewed in the web-viewer, loaded from a checkpoint, rendered, and exported.

We provide a few additional variants:

| Method           | Description                    | Memory | Speed   |
| ---------------- | ------------------------------ | ------ | ------- |
| `splatfacto`     | Default Model                  | ~6GB   | Fast    |
| `splatfacto-big` | More Gaussians, Higher Quality | ~12GB  | Slower  |


A full evalaution of Nerfstudio's implementation of Gaussian Splatting against the original Inria method can be found [here](https://docs.gsplat.studio/tests/eval.html).

#### Quality and Regularization
The default settings provided maintain a balance between speed, quality, and splat file size, but if you care more about quality than training speed or size, you can decrease the alpha cull threshold 
(threshold to delete translucent gaussians) and disable culling after 15k steps like so: `ns-train splatfacto --pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False --data <data>`

A common artifact in splatting is long, spikey gaussians. [PhysGaussian](https://xpandora.github.io/PhysGaussian/) proposes a scale regularizer that encourages gaussians to be more evenly shaped. To enable this, set the `use_scale_regularization` flag to `True`.

### Details
For more details on the method, see the [original paper](https://arxiv.org/abs/2308.04079). Additionally, for a detailed derivation of the gradients used in the gsplat library, see [here](https://arxiv.org/abs/2312.02121).

### Exporting splats
Gaussian splats can be exported as a `.ply` file which are ingestable by a variety of online web viewers. You can do this via the viewer, or `ns-export gaussian-splat --load-config <config> --output-dir exports/splat`. Currently splats can only be exported from trained splats, not from nerfacto.

Nerfstudio's splat export currently supports multiple third-party splat viewers:
- [Polycam Viewer](https://poly.cam/tools/gaussian-splatting)
- [Playcanvas SuperSplat](https://playcanvas.com/super-splat)
- [WebGL Viewer by antimatter15](https://antimatter15.com/splat/) 
- [Spline](https://spline.design/) 
- [Three.js Viewer by mkkellogg](https://github.com/mkkellogg/GaussianSplats3D)

### FAQ
- Can I export a mesh or pointcloud?

Currently these export options are not supported, but may be in the future. Contributions are always welcome!
- Can I render fisheye, equirectangular, orthographic images?

Currently, no. Gaussian rasterization assumes a perspective camera for its rasterization pipeline. Implementing other camera models is of interest but not currently planned.
## nerfology/methods

### nerfbusters.md

# Nerfbusters

<h4>Removing Ghostly Artifacts from Casually Captured NeRFs 👻 </h4>

```{button-link} https://ethanweber.me/nerfbusters
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/ethanweber/nerfbusters
:color: primary
:outline:
Code and Data
```

**TLDR: We present a method that uses a 3D diffusion prior to clean NeRFs and an evaluation procedure for in-the-wild NeRFs**

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://ethanweber.me/nerfbusters/media/nerf-comparisons/car-sq.mp4" type="video/mp4">
</video>

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/ethanweber/nerfbusters
nerfbusters-setup
ns-train nerfbusters --help
```

For more details, see the [installation instructions](https://github.com/ethanweber/nerfbusters).

### Running the Method

Please checkout the readme for the [Nerfbusters repository](https://github.com/ethanweber/nerfbusters)

## Abstract

Casually captured Neural Radiance Fields (NeRFs) suffer from artifacts such as floaters or flawed geometry when rendered outside the path of the training views. However, common practice is still to evaluate on every 8th frame, which does not measure rendering quality away from training views, hindering progress in volume rendering. We propose a new dataset and evaluation procedure, where two camera trajectories are recorded of the scene, one used for training, and the other for evaluation. We find that existing hand-crafted regularizers do not remove floaters nor improve scene geometry in this more challenging in-the-wild setting. To this end, we propose a learned, local 3D diffusion prior and a novel density score distillation sampling loss. We show that this learned prior removes floaters and improves scene geometry for casual captures.

## NeRF evaluation for casually captured videos

We train on a single camera trajectory and evaluate on a second camera trajectory. Both methods succeed on the training trajectory, but only our method succeeds on the evaluation trajectory.

<video id="teaser" muted autoplay playsinline loop no-controls width="100%">
    <source id="mp4" src="https://ethanweber.me/nerfbusters/media/teaser-animated.m4v" type="video/mp4">
</video>

### Model Overview

We learn a local 3D prior with a diffusion model that regularizes the 3D geometry of NeRFs. We use importance sampling to query a cube with NeRF densities. We binarize these densities and perform one single denoising step using a pre-trained 3D diffusion model. With these denoised densities, we compute a density score distillation sampling (DSDS) that penalizes NeRF densities where the diffusion model predicts empty voxels and pushes the NeRF densities above the target w where the diffusion model predicts occupied voxels.

![Overview](https://ethanweber.me/nerfbusters/media/method.png)


### Visibiltiy Loss

Our visibility loss enables stepping behind or outside the training camera frustums. We accomplish this by supervising densities to be low when not seen by at least one training view. Other solutions would be to store an occupancy grid or compute ray-frustum intersection tests during rendering. Our solution is easy to implement and applicable to any NeRF.

![Visibility](https://ethanweber.me/nerfbusters/media/visibility_loss.png)

### Results and dataset preview

For results and a dataset preview, view the [project page](https://ethanweber.me/nerfbusters)!
## nerfology/methods

### seathru_nerf.md

# SeaThru-NeRF
```{button-link} https://sea-thru-nerf.github.io
:color: primary
:outline:
Official Paper Website
```

```{button-link} https://github.com/AkerBP/seathru_nerf
:color: primary
:outline:
Code (nerfstudio implementation)
```

<p align="center">
  <img src="https://github.com/AkerBP/seathru_nerf/blob/main/imgs/comp.gif?raw=true" alt="Example Render">
</p>

**A Neural Radiance Field for subsea scenes.**

## Requirements

We provide the following two model configurations:

| Method              | Description   | Memory | Quality |
| ------------------- | ------------- | ------ | ------- |
| `seathru-nerf`      | Larger model  | ~23 GB | Best    |
| `seathru-nerf-lite` | Smaller model | ~7 GB  | Good    |

`seathru-nerf-lite` should run on a single desktop/laptop GPU with 8GB VRAM.

## Installation

After installing nerfstudio and its dependencies, run:

```bash
pip install git+https://github.com/AkerBP/seathru_nerf
```

## Running SeaThru-NeRF

To check your installation and to see the hyperparameters of the method, run:

```bash
ns-train seathru-nerf-lite --help
```

If you see the help message with all the training options, you are good to go and ready to train your first susbea NeRF! 🚀🚀🚀

For a detailed tutorial of a training process, please see the docs provided [here](https://akerbp.github.io/seathru_nerf/).


## Method
This method is an unofficial extension that adapts the official [SeaThru-NeRF](https://sea-thru-nerf.github.io) publication. Since it is built ontop of nerfstudio, it allows for easy modification and experimentation.

Compared to a classical NeRF approach, we differentiate between solid objects and the medium within a scene. Therefore both, the object colours and the medium colours of samples along a ray contribute towards the final pixel colour as follows:

$$\boldsymbol{\hat{C}}(\mathbf{r})=\sum_{i=1}^N \boldsymbol{\hat{C}}^{\rm obj}_i(\mathbf{r})+\sum_{i=1}^N \boldsymbol{\hat{C}}^{\rm med}_i(\mathbf{r}) \,.$$

Those two contributions can be calculated as follows:

$$\boldsymbol{\hat{C}}^{\rm obj}_i(\mathbf{r}) =
  T^{\rm obj}_i \cdot \exp (-\boldsymbol{\sigma}^{\rm attn} t_i)
  \cdot \big(1-\exp({-\sigma^{\rm obj}_i\delta_i})\big) \cdot \mathbf{c}^{\rm obj}_i \,,$$
$$\boldsymbol{\hat{C}}^{\rm med}_i(\mathbf{r}) = 
  T^{\rm obj}_i \cdot \exp ( -\boldsymbol{\sigma}^{\rm bs} t_i )
  \cdot \big( 1 - \exp ( -\boldsymbol{\sigma}^{\rm bs} \delta_i ) \big) \cdot \mathbf{c}^{\rm med}\,,$$
$$\textrm{where } \ T^{\rm obj}_i = \exp\bigg(-\sum_{j=0}^{i-1}\sigma^{\rm obj}_j\delta_j\bigg) \,. $$

The above equations contain five parameters that are used to describe the underlying scene: object density $\sigma^{\rm obj}_i \in \mathbb{R}^{1}$, object colour $\mathbf{c}^{obj}_i \in \mathbb{R}^{3}$, backscatter density $\boldsymbol{\sigma}^{\rm bs} \in \mathbb{R}^{3}$, attenuation density $\boldsymbol{\sigma}^{\rm attn} \in \mathbb{R}^{3}$ and medium colour $\mathbf{c}^{\rm med} \in \mathbb{R}^{3}$.

To get a better idea of the different densities, the following figure shows an example ray with the different densities visualised:

<p align="center">
<img src="https://github.com/AkerBP/seathru_nerf/blob/main/imgs/ray.png?raw=true" width=60% alt="SeaThru-NeRF ray">
</p>

*The image above was taken from [Levy et al. (2023)](https://arxiv.org/abs/2304.07743).*

To predict the object and medium parameters, we use two separate networks. This subsea specific approach can be visualised as follows: (note that the third network is the proposal network, which is used to sample points in regions that contribute most to the final image)

<p align="center">
<img src="https://github.com/AkerBP/seathru_nerf/blob/main/imgs/architecture.png?raw=true" width=60% alt="SeaThru-NeRF Architecture">
</p>

If you are interested in an in depth description of the method, make sure to check out the documentation [here](https://akerbp.github.io/seathru_nerf/intro.html).


## Example results
Due to the underlying image formation model that allows us to seperate between the objects and the water within a scene, you can choose between different rendering options. The following options exist:

- rgb: To render normal RGB of the scene.
- J: To render the clear scene (water effect removed).
- direct: To render the attenuated clear scene.
- bs: To render backscatter of the water within the scene.
- depth: To render depthmaps of the scene.
- accumulation: To render object weight accumulation of the scene.

Below, you can see an original render of a scene and one with the water effects removed:

<p align="center">
<img src="https://github.com/AkerBP/seathru_nerf/blob/main/imgs/example_render_rgb.gif?raw=true" alt="RBG rendering"/>
<img src="https://github.com/AkerBP/seathru_nerf/blob/main/imgs/example_render_J.gif?raw=true" alt="J rendering"/>
</p>

*Please note that those gifs are compressed and do not do the approach justice. Please render your own videos to see the level of detail and clarity of the renders.*## nerfology/methods

### nerf.md

# NeRF

<h4>Neural Radiance Fields</h4>

```{button-link} https://www.matthewtancik.com/nerf
:color: primary
:outline:
Paper Website
```

### Running the model

```bash
ns-train vanilla-nerf
```

## Method overview

If you have arrived to this site, it is likely that you have at least heard of NeRFs. This page will discuss the original NeRF paper, _"NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"_ by Mildenhall, Srinivasan, Tancik et al. (2020). 

For most tasks, using the original NeRF model is likely not a good choice and hence we provide implementations of various other NeRF related models. It is however useful to understand how NeRF's work as most follow ups follow a similar structure and it doesn't require CUDA to execute (useful for stepping through the code with a debugger if you don't have a GPU at hand).

The goal is to optimize a volumetric representation of a scene that can be rendered from novel viewpoints. This representation is optimized from a set of images and associated camera poses.

```{admonition} Assumptions
If any of the following assumptions are broken, the reconstructions may fail completely or contain artifacts such as excess geometry.
* Camera poses are known
* Scene is static, objects do not move
* The scene appearance is constant (ie. exposure doesn't change)
* Dense input capture (Each point in the scene should be visible in multiple images)
```

## Pipeline

```{image} imgs/models_nerf-pipeline-light.png
:align: center
:class: only-light
```

```{image} imgs/models_nerf-pipeline-dark.png
:align: center
:class: only-dark
```

Here is an overview pipeline for NeRF, we will walk through each component in this guide.

### Field representation

```{image} imgs/models_nerf-pipeline-field-light.png
:align: center
:class: only-light
```

```{image} imgs/models_nerf-pipeline-field-dark.png
:align: center
:class: only-dark
```

NeRFs are a volumetric representation encoded into a neural network. They are not 3D meshes and they are not voxels. **For each point in space the NeRF represents a view dependent radiance.** More concretely each point has a density which describes how transparent or opaque a point in space is. They also have a view dependent color that changes depending on the angle the point is viewed.

```{image} imgs/models_nerf-field-light.png
:align: center
:class: only-light
:width: 400
```

```{image} imgs/models_nerf-field-dark.png
:align: center
:class: only-dark
:width: 400
```

The associated NeRF fields can be instantiated with the following nerfstudio code (encoding described in next section):

```python
from nerfstudio.fields.vanilla_nerf_field import NeRFField

field_coarse = NeRFField(position_encoding=pos_enc, direction_encoding=dir_enc)
field_fine = NeRFField(position_encoding=pos_enc, direction_encoding=dir_enc)
```

#### Positional encoding

An extra trick is necessary to make the neural network expressive enough to represent fine details in the scene. The input coordinates $(x,y,z,\theta,\phi)$ need to be encoded to a higher dimensional space prior to being input into the network. You can learn more about encodings [here](../model_components/visualize_encoders.ipynb).

```python
from nerfstudio.field_components.encodings import NeRFEncoding

pos_enc = NeRFEncoding(
    in_dim=3, num_frequencies=10, min_freq_exp=0.0, max_freq_exp=8.0, include_input=True
)
dir_enc = NeRFEncoding(
    in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=4.0, include_input=True
)
```

### Rendering

```{image} imgs/models_nerf-pipeline-renderer-light.png
:align: center
:class: only-light
```

```{image} imgs/models_nerf-pipeline-renderer-dark.png
:align: center
:class: only-dark
```

Now that we have a representation of space, we need some way to render new images of it. To accomplish this, we are going to _project_ a ray from the target pixel and evaluate points along that ray. We then rely on classic volumetric rendering techniques [[Kajiya, 1984]](https://dl.acm.org/doi/abs/10.1145/964965.808594) to composite the points into a predicted color. 

This compositing is similar to what happens in tools like Photoshop when you layer multiple objects of varying opacity on top of each other. The only difference is that NeRF takes into account the differences in spacing between points.

Rending RGB images is not the only type of output render supported. It is possible to render other output types such as depth and semantics. Additional renderers can be found [Here](../../reference/api/model_components/renderers.rst).

Associated nerfstudio code:

```python
from nerfstudio.model_components.renderers import RGBRenderer

renderer_rgb = RGBRenderer(background_color=colors.WHITE)
# Ray samples discussed in the next section
field_outputs = field_coarse.forward(ray_samples)
weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
rgb = renderer_rgb(
    rgb=field_outputs[FieldHeadNames.RGB],
    weights=weights,
)
```

#### Sampling

```{image} imgs/models_nerf-pipeline-sampler-light.png
:align: center
:class: only-light
```

```{image} imgs/models_nerf-pipeline-sampler-dark.png
:align: center
:class: only-dark
```

How we sample points along rays in space is an important design decision. Various sampling strategies can be used which are discussed in detail in the [Ray Samplers](../model_components/visualize_samplers.ipynb) guide. In NeRF we take advantage of a hierarchical sampling scheme that first uses a _uniform sampler_ and is followed by a _PDF sampler_. 

The uniform sampler distributes samples evenly between a predefined distance range from the camera. These are then used to compute an initial render of the scene. The renderer optionally produces _weights_ for each sample that correlate with how important each sample was to the final renderer. 

The PDF sampler uses these _weights_ to generate a new set of samples that are biased to regions of higher weight. In practice, these regions are near the surface of the object.

Associated code:

```python
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler

sampler_uniform = UniformSampler(num_samples=num_coarse_samples)
ray_samples_uniform = sampler_uniform(ray_bundle)

sampler_pdf = PDFSampler(num_samples=num_importance_samples)
field_outputs_coarse = field_coarse.forward(ray_samples_uniform)
weights_coarse = ray_samples_uniform.get_weights(field_outputs_coarse[FieldHeadNames.DENSITY])
ray_samples_pdf = sampler_pdf(ray_bundle, ray_samples_uniform, weights_coarse)
```

```{warning}
Described above is specific to scenes that have known bounds (ie. the Blender Synthetic dataset). For unbounded scenes, the original NeRF paper uses Normalized Device Coordinates (NDC) to warp space, along with a _linear in disparity_ sampler. We do not support NDC, for unbounded scenes consider using [Spatial Distortions](../model_components/visualize_spatial_distortions.ipynb).
```

```{tip}
For all sampling, we use _Stratified_ samples during optimization and unmodified samples during inference. Further details can be found in the [Ray Samplers](../model_components/visualize_samplers.ipynb) guide.
```

## Benchmarks

##### Blender synthetic

| Implementation                                                                    |    Mic    | Ficus     |   Chair   | Hotdog    | Materials | Drums     | Ship      | Lego      | Average   |
| --------------------------------------------------------------------------------- | :-------: | --------- | :-------: | --------- | --------- | --------- | --------- | --------- | --------- |
| nerfstudio                                                                        |   33.76   | **31.98** | **34.35** | 36.57     | **31.00** | **25.11** | 29.87     | **34.46** | **32.14** |
| [TF NeRF](https://github.com/bmild/nerf)                                          |   32.91   | 30.13     |   33.00   | 36.18     | 29.62     | 25.01     | 28.65     | 32.54     | 31.04     |
| [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) | **34.53** | 30.43     |   34.08   | **36.92** | 29.91     | 25.03     | **29.36** | 33.28     | 31.69     |
## nerfology/methods

### index.md

# Methods

We provide a set of pre implemented nerfstudio methods.

**The goal of nerfstudio is to modularize the various NeRF techniques as much as possible.**

As a result, many of the techniques from these pre-implemented methods can be mixed 🎨.

## Running a method

It's easy!

```bash
ns-train {METHOD_NAME}
```

To list the available methods run:

```bash
ns-train --help
```

## Implemented Methods

The following methods are supported in nerfstudio:

```{toctree}
    :maxdepth: 1
    Instant-NGP<instant_ngp.md>
    Splatfacto<splat.md>
    Instruct-NeRF2NeRF<in2n.md>
    K-Planes<kplanes.md>
    LERF<lerf.md>
    Mip-NeRF<mipnerf.md>
    NeRF<nerf.md>
    Nerfacto<nerfacto.md>
    Nerfbusters<nerfbusters.md>
    NeRFPlayer<nerfplayer.md>
    Tetra-NeRF<tetranerf.md>
    TensoRF<tensorf.md>
    Generfacto<generfacto.md>
    Instruct-GS2GS<igs2gs.md>
    PyNeRF<pynerf.md>
    SeaThru-NeRF<seathru_nerf.md>
    Zip-NeRF<zipnerf.md>
```

(own_method_docs)=

## Adding Your Own Method

If you're a researcher looking to develop new NeRF-related methods, we hope that you find nerfstudio to be a useful tool. We've provided documentation about integrating with the nerfstudio codebase, which you can find [here](../../developer_guides/new_methods.md).

We also welcome additions to the list of methods above. To do this, simply create a pull request with the following changes,

1. Add a markdown file describing the model to the `docs/nerfology/methods` folder
2. Update the above list of implement methods in this file.
3. Add the method to {ref}`this<third_party_methods>` list in `docs/index.md`.
4. Add a new `ExternalMethod` entry to the `nerfstudio/configs/external_methods.py` file.

For the method description, please refer to the [Instruct-NeRF2NeRF](in2n) page as an example of the layout. Please try to include the following information:

- Installation instructions
- Instructions for running the method
- Requirements (dataset, GPU, ect)
- Method description (the more detailed the better, treat it like a blog post)

You are welcome to include assets (such as images or video) in the description, but please host them elsewhere.

:::{admonition} Note
:class: note

Please ensure that the documentation is clear and easy to understand for other users who may want to try out your method.
:::
## nerfology/methods

### zipnerf.md

# Zip-NeRF

<h4>A pytorch implementation of "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields"</h4>

```{button-link} https://jonbarron.info/zipnerf/
:color: primary
:outline:
Paper Website
```
```{button-link} https://github.com/SuLvXiangXin/zipnerf-pytorch
:color: primary
:outline:
Code
```
### Installation
First, install nerfstudio and its dependencies. Then run:
```
pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch#subdirectory=extensions/cuda
pip install git+https://github.com/SuLvXiangXin/zipnerf-pytorch
```
Finally, install torch_scatter corresponding to your cuda version(https://pytorch-geometric.com/whl/torch-2.0.1%2Bcu118.html).


### Running Model

```bash
ns-train zipnerf --data {DATA_DIR/SCENE}
```

## Overview
Zipnerf combines mip-NeRF 360’s overall framework with iNGP’s featurization approach.
Following mip-NeRF, zipnerf assume each pixel corresponds to a cone. Given an interval along the ray, it construct a set of multisamples that approximate the shape of that conical frustum.
Also,  it present an alternative loss that, unlike mip-NeRF 360’s interlevel loss, is continuous and smooth with respect to distance along the ray to prevent z-aliasing.## nerfology/methods

### nerfacto.md

# Nerfacto

<h4>Our *defacto* method.</h4>
 
### Running the Method
 
```bash
ns-train nerfacto --help
```

We provide a few additional variants:

| Method           | Description                    | Memory | Speed   |
| ---------------- | ------------------------------ | ------ | ------- |
| `nerfacto`       | Default Model                  | ~6GB   | Fast    |
| `nerfacto-big`   | Larger higher quality          | ~12GB  | Slower  |
| `nerfacto-huge`  | Even larger and higher quality | ~24GB  | Slowest |
| `depth-nerfacto` | Supervise on depth             | ~6GB   | Fast    |

## Method

### Overview

We created the nerfacto model to act as our default for real data captures of static scenes. The model is not existing published work, but rather a combination of many published methods that we have found work well for real data. This guide discusses the details of the model, understanding the [NeRF model](./nerf.md) is a prerequisite.

```{admonition} TLDR
We combine the following techniques in this model:
* Camera pose refinement
* Per image appearance conditioning
* Proposal sampling
* Scene contraction
* Hash encoding
```

```{warning}
🏗️ This guide is under construction 🏗️
```

## Pipeline

```{image} imgs/nerfacto/models_nerfacto_pipeline-light.png
:align: center
:class: only-light
```

```{image} imgs/nerfacto/models_nerfacto_pipeline-dark.png
:align: center
:class: only-dark
```

Here is an overview pipeline for nerfacto, we will walk through each component in this guide.

### Pose Refinement

It is not uncommon to have errors in the predicted camera poses. This is even more of a factor when using poses acquired from devices such as phones (ie. if you use the Record3D IOS app to capture data). Misaligned poses result in both cloudy artifacts in the scene and a reduction of sharpness and details. The NeRF framework allows us to backpropagate loss gradients to the input pose calculations. We can use this information to optimize and refine the poses.

### Piecewise Sampler

We use a Piecewise sampler to produce the initial set of samples of the scene. This sampler allocates half of the samples uniformly up to a distance of 1 from the camera. The remaining samples are distributed such that the step size increases with each sample. The step size is chosen such that the [frustums](../model_components/visualize_samples.ipynb) are scaled versions of themselves. By increasing the step sizes, we are able to sample distant objects while still having a dense set of samples for nearby objects.

### Proposal Sampler

The proposal sampler consolidates the sample locations to the regions of the scene that contribute most to the final render (typically the first surface intersection). This greatly improves reconstruction quality. The proposal network sampler requires a density function for the scene. The density function can be implemented in a variety of ways, we find that using a small fused-mlp with a hash encoding has sufficient accuracy and is fast. The proposal network sampler can be chained together with multiple density functions to further consolidate the sampling. We have found that two density functions are better than one. Larger than 2 leads to diminishing returns.

#### Density Field

The density field only needs to represent a coarse density representation of the scene to guide sampling. Combining a hash encoding with a small fused MLP (from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)) provides a fast way to query the scene. We can make it more efficient by decreasing the encoding dictionary size and number of feature levels. These simplifications have little impact on the reconstruction quality because the density function does not need to learn high frequency details during the initial passes.

### Nerfacto Field

```{image} imgs/nerfacto/models_nerfacto_field-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/nerfacto/models_nerfacto_field-dark.png
:align: center
:class: only-dark
:width: 600
```

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/models/nerfacto.py
:color: primary
:outline:
See the code!
```
## nerfology/methods

### generfacto.md

# Generfacto

<h4>Generate 3D models from text</h4>

**Our model that combines generative 3D with our latest NeRF methods**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install -e .[gen]
```

Two options for text to image diffusion models are provided: Stable Diffusion and DeepFloyd IF.
We use Deepfloyd IF by default because it trains faster and produces better results. Using this model requires users to sign a license agreement for the model card of DeepFloyd IF, which can be found [here](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). Once the licensed is signed, log into Huggingface locally by running the following command:

```bash
huggingface-cli login
```

If you do not want to sign the license agreement, you can use the Stable Diffusion model (instructions below).

## Running Generfacto

Once installed, run:

```bash
ns-train generfacto --prompt "a high quality photo of a pineapple"
```

The first time you run this method, the diffusion model weights will be downloaded and cached
from Hugging Face, which may take a couple minutes.

Specify which diffusion model to use with the diffusion_model flag:

```bash
ns-train generfacto --pipeline.model.diffusion_model ["stablediffusion", "deepfloyd"]
```

## Example Results

The following videos are renders of NeRFs generated from Generfacto. Each model was trained 30k steps, which took around 1 hour with DeepFloyd
and around 4 hours with Stable Diffusion.

"a high quality photo of a ripe pineapple" (Stable Diffusion)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646597-407ff7c8-7106-4835-acf3-c2f8188bbd1d.mp4" type="video/mp4">
</video>

"a high quality zoomed out photo of a palm tree" (DeepFloyd)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646594-05ffebce-a3d6-43af-9f11-e04ce2ce3237.mp4" type="video/mp4">
</video>

"a high quality zoomed out photo of a light grey baby shark" (DeepFloyd)

<video id="idu" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://user-images.githubusercontent.com/19509183/246646599-b1f5b7c5-dd96-48b4-8db0-960632e7798b.mp4" type="video/mp4">
</video>
## nerfology/methods

### igs2gs.md

# Instruct-GS2GS

<h4>Editing Gaussian Splatting Scenes with Instructions</h4>

```{button-link} https://instruct-gs2gs.github.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/cvachha/instruct-gs2gs
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/face.mp4" type="video/mp4">
</video>

**Instruct-GS2GS enables instruction-based editing of 3D Gaussian Splatting scenes via a 2D diffusion model**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/cvachha/instruct-gs2gs
cd instruct-gs2gs
pip install --upgrade pip setuptools
pip install -e .
```

## Running Instruct-GS2GS

Details for running Instruct-GS2GS (built with Nerfstudio!) can be found [here](https://github.com/cvachha/instruct-gs2gs). Once installed, run:

```bash
ns-train igs2gs --help
```

| Method       | Description                  | Memory |
| ------------ | ---------------------------- | ------ |
| `igs2gs`       | Full model, used in paper    | ~15GB  |

Datasets need to be processed with COLMAP for Gaussian Splatting support.

Once you have trained your GS scene for 20k iterations, the checkpoints will be saved to the `outputs` directory. Copy the path to the `nerfstudio_models` folder. (Note: We noticed that training for 20k iterations rather than 30k seemed to run more reliably)

To start training for editing the GS, run the following command:

```bash
ns-train igs2gs --data {PROCESSED_DATA_DIR} --load-dir {outputs/.../nerfstudio_models} --pipeline.prompt {"prompt"} --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5
```

The `{PROCESSED_DATA_DIR}` must be the same path as used in training the original GS. Using the CLI commands, you can choose the prompt and the guidance scales used for InstructPix2Pix.

## Method

### Overview

Instruct-GS2GS is a method for editing 3D Gaussian Splatting (3DGS) scenes with text instructions in a method based on [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/). Given a 3DGS scene of a scene and the collection of images used to reconstruct it, this method uses an image-conditioned diffusion model ([InstructPix2Pix](https://www.timothybrooks.com/instruct-pix2pix)) to iteratively edit the input images while optimizing the underlying scene, resulting in an optimized 3D scene that respects the edit instruction. The paper demonstrates that our proposed method is able to edit large-scale, real-world scenes, and is able to accomplish  realistic and targeted edits.


## Pipeline

<video id="pipeline" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/pipeline.mp4" type="video/mp4">
</video>

This section will walk through each component of the Instruct-GS2GS method.

### How it Works

Instruct-GS2GS gradually updates a reconstructed Gaussian Splatting scene by iteratively updating the dataset images while training the 3DGS:

1. Images are rendered from the scene at all training viewpoints.
2. They get edited by InstructPix2Pix given a global text instruction.
3. The training dataset images are replaced with the edited images.
4. The 3DGS continues training as usual for 2.5k iterations.

### Editing Images with InstructPix2Pix

To update a dataset image from a given viewpoint, Instruct-GS2GS takes the original, unedited training image as image conditioning and uses the global text instruction as text conditioning. This process mixes the information of the diffusion model, which attempts to edit the image, the current 3D structure of the 3DGS, and view-consistent information from the unedited, ground-truth images. By combining this set of information, the edit is respected while maintaining 3D consistency.

The code snippet for how an image is edited in the pipeline can be found [here](https://github.com/cvachha/instruct-gs2gs/blob/main/igs2gs/ip2p.py).

### Iterative Dataset Update and Implementation

The method takes in a dataset of camera poses and training images, a trained 3DGS scene, and a user-specified text-prompt instruction, e.g. “make him a marble statue”. Instruct-GS2GS constructs the edited GS scene guided by the text-prompt by applying a 2D text and image conditioned diffusion model, in this case Instruct-Pix2Pix, to all training images over the course of training. It performs these edits using an iterative udpate scheme in which all training dataset images are updated using a diffusion model individually, for sequential iterations spanning the size of the training images, every 2.5k training iterations. This process allows the GS to have a holistic edit and maintain 3D consistency.

The process is similar to Instruct-NeRF2NeRF where for a given training camera view, it sets the original training image as the conditioning image, the noisy image input as the GS rendered from the camera combined with some randomly selected noise, and receives an edited image respecting the text conditioning. With this method, it is able to propagate the edited changes to the GS scene. The method is able to maintain grounded edits by conditioning Instruct-Pix2Pix on the original unedited training image.

This method uses Nerfstudio’s gsplat library for our underlying gaussian splatting model. We adapt similar parameters for the diffusion model from Instruct-NeRF2NeRF. Among these are the values that define the amount of noise (and therefore the amount signal retained from the original images). We vary the classifier-free guidance scales per edit and scene, using a range of values. We edit the entire dataset and then train the scene for 2.5k iterations. For GS training, we use L1 and LPIPS losses. We train our method for a maximum of 27.5k iterations (starting with a GS scene trained for 20k iterations). However, in practice we stop training once the edit has converged. In many cases, the optimal training length is a subjective decision — a user may prefer more subtle or more extreme edits that are best found at different stages of training.


## Results

For results, view the [project page](https://instruct-gs2gs.github.io/)!

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://instruct-gs2gs.github.io/data/videos/campanile_all.mp4" type="video/mp4">
</video>## nerfology/methods

### kplanes.md

# K-Planes

<h4>Explicit Radiance Fields in Space, Time, and Appearance</h4>


```{button-link} https://sarafridov.github.io/K-Planes/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/sarafridov/K-Plane
:color: primary
:outline:
Official Code
```

```{button-link} https://github.com/Giodiro/kplanes_nerfstudio
:color: primary
:outline:
Nerfstudio add-on code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/small_teaser.mp4" type="video/mp4">
</video>

**A unified model for static, dynamic and variable appearance NeRFs.**


## Installation

First, install nerfstudio and its dependencies. Then install the K-Planes add-on
```
pip install kplanes-nerfstudio
```

## Running K-Planes

There are two default configurations provided which use the blender and DNeRF dataloaders. However, you can easily extend them to create a new configuration for different datasets.

The default configurations provided are
| Method            | Description              | Scene type                     | Memory |
| ----------------- | -------------------------| ------------------------------ | ------ |
| `kplanes`         | Tuned for blender scenes | static, synthetic              | ~4GB   |
| `kplanes-dynamic` | Tuned for DNeRF dataset  | dynamic (monocular), synthetic | ~5GB   |


for training with these two configurations you should run
```bash
ns-train kplanes --data <data-folder>
```
or
```bash
ns-train kplanes-dynamic --data <data-folder>
```

:::{admonition} Note
:class: warning

`kplanes` is set up to use blender data, (download it running `ns-download-data blender`), 
`kplanes-dynamic` is set up to use DNeRF data, (download it running `ns-download-data dnerf`).
:::


## Method

![method overview](https://sarafridov.github.io/K-Planes/assets/intro_figure.jpg)<br>
K-planes represents a scene in k dimensions -- where k can be 3 for static 3-dimensional scenes or 4 for scenes which change in time -- 
using k-choose-2 planes (or grids). After ray-sampling, the querying the field at a certain position consists in querying each plane (with interpolation), and combining the resulting features through multiplication.
This factorization of space and time keeps memory usage low, and is very flexible in the kinds of priors and regularizers that can be added.<br>
<br>

We support hybrid models with a small MLP (left) and fully explicit models (right), through the `linear_decoder` [configuration key](https://github.com/Giodiro/kplanes_nerfstudio/blob/db4130605159dadaf180228be5d0335d2c4d21f9/kplanes/kplanes.py#L87)
<br>
<video id="4d_dynamic_mlp" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_mlp.mp4" type="video/mp4">
</video>
<video id="4d_dynamic_linear" muted autoplay playsinline loop controls width="48%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_salmon_path_linear.mp4" type="video/mp4">
</video>

The model also supports decomposing a scene into its space and time components. For more information on how to do this see the [official code repo](https://github.com/sarafridov/K-Plane)
<br>
<video id="4d_spacetime"  muted autoplay playsinline loop controls width="96%">
    <source id="mp4" src="https://sarafridov.github.io/K-Planes/assets/dynerf/small_cutbeef_spacetime_mlp.mp4" type="video/mp4">
</video>
## nerfology/methods

### pynerf.md

# PyNeRF

<h4>Pyramidal Neural Radiance Fields</h4>


```{button-link} https://haithemturki.com/pynerf/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/hturki/pynerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://haithemturki.com/pynerf/vids/ficus.mp4" type="video/mp4">
</video>

**A fast NeRF anti-aliasing strategy.**


## Installation

First, install Nerfstudio and its dependencies. Then install the PyNeRF extension and [torch-scatter](https://github.com/rusty1s/pytorch_scatter):
```
pip install git+https://github.com/hturki/pynerf
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA}.html
```

## Running PyNeRF

There are three default configurations provided which use the MipNeRF 360 and Multicam dataparsers by default. You can easily use other dataparsers via the ``ns-train`` command (ie: ``ns-train pynerf nerfstudio-data --data <your data dir>`` to use the Nerfstudio data parser)

The default configurations provided are:

| Method                  | Description                                       | Scene type                     | Memory |
| ----------------------- |---------------------------------------------------| ------------------------------ |--------|
| `pynerf `               | Tuned for outdoor scenes, uses proposal network   | outdoors                       | ~5GB   |
| `pynerf-synthetic`      | Tuned for synthetic scenes, uses proposal network | synthetic                      | ~5GB   |
| `pynerf-occupancy-grid` | Tuned for Multiscale blender, uses occupancy grid | synthetic                      | ~5GB   |


The main differences between them is whether they are suited for synthetic/indoor or real-world unbounded scenes (in case case appearance embeddings and scene contraction are enabled), and whether sampling is done with a proposal network (usually better for real-world scenes) or an occupancy grid (usally better for single-object synthetic scenes like Blender).

## Method

Most NeRF methods assume that training and test-time cameras capture scene content from a roughly constant distance:

<table>
    <tbody>
        <tr>
            <td style="width: 48%;">
                <div style="display: flex; justify-content: center; align-items: center;">
                    <img src="https://haithemturki.com/pynerf/images/ficus-cameras.jpg">
                </div>
            </td>
            <td style="width: 4%;"><img src="https://haithemturki.com/pynerf/images/arrow-right-white.png" style="width: 100%;"></td>
            <td style="width: 48%;">
                <video width="100%" autoplay loop controls>
                    <source src="https://haithemturki.com/pynerf/vids/ficus-rotation.mp4" type="video/mp4" poster="https://haithemturki.com/pynerf/images/ficus-rotation.jpg">
                </video>
            </td>
        </tr>
    </tbody>
</table>

They degrade and render blurry views in less constrained settings:

<table>
    <tbody>
        <tr>
            <td style="width: 48%;">
                <div style="display: flex; justify-content: center; align-items: center;">
                    <img src="https://haithemturki.com/pynerf//images/ficus-cameras-different.jpg">
                </div>
            </td>
            <td style="width: 4%;"><img src="https://haithemturki.com/pynerf/images/arrow-right-white.png" style="width: 100%;"></td>
            <td style="width: 48%;">
                <video width="100%" autoplay loop controls>
                    <source src="https://haithemturki.com/pynerf/vids/ficus-zoom-nerf.mp4" type="video/mp4" poster="https://haithemturki.com/pynerf/images/ficus-zoom-nerf.jpg">
                </video>
            </td>
        </tr>
    </tbody>
</table>

This is due to NeRF being scale-unaware, as it reasons about point samples instead of volumes. We address this by training a pyramid of NeRFs that divide the scene at different resolutions. We use "coarse" NeRFs for far-away samples, and finer NeRF for close-up samples:

<img src="https://haithemturki.com/pynerf/images/model.jpg" width="70%" style="display: block; margin-left: auto; margin-right: auto">## nerfology/methods

### mipnerf.md

# Mip-NeRF

<h4>A Multiscale Representation for Anti-Aliasing Neural Radiance Fields</h4>

```{button-link} https://jonbarron.info/mipnerf/
:color: primary
:outline:
Paper Website
```

### Running Model

```bash
ns-train mipnerf
```

## Overview

```{image} imgs/mipnerf/models_mipnerf_pipeline-light.png
:align: center
:class: only-light
```

```{image} imgs/mipnerf/models_mipnerf_pipeline-dark.png
:align: center
:class: only-dark
```

The primary modification in MipNeRF is in the encoding for the field representation. With the modification the same _mip-NeRF_ field can be use for the coarse and fine steps of the rendering hierarchy.

```{image} imgs/mipnerf/models_mipnerf_field-light.png
:align: center
:class: only-light
:width: 400
```

```{image} imgs/mipnerf/models_mipnerf_field-dark.png
:align: center
:class: only-dark
:width: 400
```

In the field, the Positional Encoding (PE) is replaced with an Integrated Positional Encoding (IPE) that takes into account the size of the sample.
## nerfology/methods

### instant_ngp.md

# Instant-NGP

<h4>Instant Neural Graphics Primitives with a Multiresolution Hash Encoding</h4>

```{button-link} https://nvlabs.github.io/instant-ngp/
:color: primary
:outline:
Paper Website
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://nvlabs.github.io/instant-ngp/assets/teaser.mp4" type="video/mp4">
</video>

### Running the Model

Instant-NGP is built locally into Nerfstudio. To use the method, run

```bash
ns-train instant-ngp --help
```

Many of the main contributions of Instant-NGP are built into our Nerfacto method, so for real-world scenes, we recommend using the Nerfacto model.

## Method

### Overview

Instant-NGP breaks NeRF training into 3 pillars and proposes improvements to each to enable real-time training of NeRFs. The 3 pillars are:

1. An improved training and rendering algorithm via a ray marching scheme which uses an occupancy grid
2. A smaller, fully-fused neural network
3. An effective multi-resolution hash encoding, the main contribution of this paper.

The core idea behind the improved sampling technique is that sampling over empty space should be skipped and sampling behind high density areas should also be skipped. This is achieved by maintaining a set of multiscale occupancy grids which coarsely mark empty and non-empty space. Occupancy is stored as a single bit, and a sample on a ray is skipped if its occupancy is too low. These occupancy grids are stored independently of the trainable encoding and are updated throughout training based on the updated density predictions. The authors find they can increase sampling speed by 10-100x compared to naive approaches.

Nerfstudio uses [NerfAcc](https://www.nerfacc.com/index.html) as the sampling algorithm implementation. The details on NerfAcc's sampling and occupancy grid is discussed [here](https://www.nerfacc.com/en/stable/methodology/sampling.html#occupancy-grid-estimator).

Another major bottleneck for NeRF's training speed has been querying the neural network. The authors of this work implement the network such that it runs entirely on a single CUDA kernel. The network is also shrunk down to be just 4 layers with 64 neurons in each layer. They show that their fully-fused neural network is 5-10x faster than a Tensorflow implementation.

Nerfstudio uses the [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) framework to utilize the fully-fused neural networks.

The speedups at each level are multiplicative. With all their improvements, Instant-NGP reaches speedups of 1000x, which enable training NeRF scenes in a matter of seconds!

### Multi-Resolution Hash Encoding

```{image} imgs/ingp/hash_figure.png
:align: center
```

One contribution of Instant-NGP is the multi-resolution hash encoding. In the traditional NeRF pipelines, input coordinates are mapped to a higher dimensional space using a positional encoding function, which is described [here](../model_components/visualize_encoders.ipynb). Instant-NGP proposes a trainable hash-based encoding. The idea is to map coordinates to trainable feature vectors which can be optimized in the standard flow of NeRF training.

The trainable features are F-dimensional vectors and are arranged into L grids which contain up to T vectors, where L represents the number of resolutions for features and T represents the number of feature vectors in each hash grid. The steps for the hash grid encoding, as shown in the figure provided by the authors, are as follows:

1. Given an input coordinate, find the surrounding voxels at L resolution levels and hash the vertices of these grids.
2. The hashed vertices are used as keys to look up trainable F-dimensional feature vectors.
3. Based on where the coordinate lies in space, the feature vectors are linearly interpolated to match the input coordinate.
4. The feature vectors from each grid are concatenated, along with any other parameters such as viewing direction,
5. The final vector is inputted into the neural network to predict the RGB and density output.

Steps 1-3 are done independently at each resolution level. Thus, since these feature vectors are trainable, when backpropagating the loss gradient, the gradients will flow through the neural network and interpolation function all the way back to the feature vectors. The feature vectors are interpolated relative to the coordinate such that the network can learn a smooth function.

An important note is that hash collisions are not explicitly handled. At each hash index, there may be multiple vertices which index to that feature vector, but because these vectors are trainable, the vertices that are most important to the specific output will have the highest gradient, and therefore automatically dominate the optimization of that feature.

This encoding structure creates a tradeoff between quality, memory, and performance. The main parameters which can be adjusted are the size of the hash table (T), the size of the feature vectors (F), and the number of resolutions (L).

Instant-NGP encodes the viewing direction using spherical harmonic encodings.

Our [`nerfacto`](./nerfacto.md) model uses both the fully-fused MLP and the hash encoder, which were inspired by Instant-NGP. Lastly, our implementation covers the major ideas from Instant-NGP, but it doesn't strictly follow every detail. Some known differences include learning rate schedulers, hyper-parameters for sampling, and how camera gradients are calculated if enabled.
## nerfology/methods

### nerfplayer.md

# NeRFPlayer

<h4>A Streamable Dynamic Scene Representation with Decomposed Neural Radiance Fields</h4>


```{button-link} https://lsongx.github.io/projects/nerfplayer.html
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/lsongx/nerfplayer-nerfstudio
:color: primary
:outline:
Nerfstudio add-on code
```

[![NeRFPlayer Video](https://img.youtube.com/vi/flVqSLZWBMI/0.jpg)](https://www.youtube.com/watch?v=flVqSLZWBMI)


## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/lsongx/nerfplayer-nerfstudio.git
```

## Running NeRFPlayer

Details for running NeRFPlayer can be found [here](https://github.com/lsongx/nerfplayer-nerfstudio). Once installed, run:

```bash
ns-train nerfplayer-ngp --help
```

Two variants of NeRFPlayer are provided:

| Method                | Description                                     |
| --------------------- | ----------------------------------------------- |
| `nerfplayer-nerfacto` | NeRFPlayer with nerfacto backbone               |
| `nerfplayer-ngp`      | NeRFPlayer with instant-ngp-bounded backbone    |


## Method Overview

![method overview](https://lsongx.github.io/projects/images/nerfplayer-framework.png)<br>
First, we propose to decompose the 4D spatiotemporal space according to temporal characteristics. Points in the 4D space are associated with probabilities of belonging to three categories: static, deforming, and new areas. Each area is represented and regularized by a separate neural field. Second, we propose a hybrid representations based feature streaming scheme for efficiently modeling the neural fields.

Please see [TODO lists](https://github.com/lsongx/nerfplayer-nerfstudio#known-todos) to see the unimplemented components in the nerfstudio based version.## nerfology/methods

### tensorf.md

# TensoRF

<h4>Tensorial Radiance Fields</h4>

```{button-link} https://apchenstu.github.io/TensoRF/
:color: primary
:outline:
Paper Website
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://apchenstu.github.io/TensoRF/video/train_process.mp4" type="video/mp4">
</video>

### Running Model

```bash
ns-train tensorf
```

## Overview

```{image} imgs/tensorf/models_tensorf_pipeline.png
:align: center
```

TensoRF models the radiance field of a scene as a 4D tensor, which represents a 3D voxel grid with per-voxel multi-channel features. TensoRF factorizes the 4D scene tensor into multiple compact low-rank tensor components using CP or VM modes. CP decomposition factorizes tensors into rank-one components with compact vectors. Vector-Matrix (VM) decomposition factorizes tensors into compact vector and matrix factors.

```{image} imgs/tensorf/models_tensorf_factorization.png
:align: center
```

TensoRF with CP(left) and VM(right) decompositions results in a significantly reduced memory footprint compared to previous and concurrent works that directly optimize per-voxel features, such as [Plenoxels](https://alexyu.net/plenoxels/) and [PlenOctrees](https://alexyu.net/plenoctrees/). In experiments, TensoRF with CP decomposition achieves fast reconstruction with improved rendering quality and a smaller model size compared to NeRF. Furthermore, TensoRF with VM decomposition enhances rendering quality even further, while reducing reconstruction time and maintaining a compact model size.
## nerfology/methods

### lerf.md

# LERF

<h4>📎 Language Embedded Radiance Fields 🚜</h4>

```{button-link} https://www.lerf.io/
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/kerrj/lerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://www.lerf.io/data/teaser.mp4" type="video/mp4">
</video>

**Grounding CLIP vectors volumetrically inside NeRF allows flexible natural language queries in 3D**

## Installation

First install nerfstudio dependencies. Then run:

```bash
pip install git+https://github.com/kerrj/lerf
```

## Running LERF

Details for running LERF (built with Nerfstudio!) can be found [here](https://github.com/kerrj/lerf). Once installed, run:

```bash
ns-train lerf --help
```

Three variants of LERF are provided:

| Method      | Description                                     | Memory | Quality |
| ----------- | ----------------------------------------------- | ------ | ------- |
| `lerf-big`  | LERF with OpenCLIP ViT-L/14                     | ~22 GB | Best    |
| `lerf`      | Model with OpenCLIP ViT-B/16, used in paper     | ~15 GB | Good    |
| `lerf-lite` | LERF with smaller network and less LERF samples | ~8 GB  | Ok      |

`lerf-lite` should work on a single NVIDIA 2080.
`lerf-big` is experimental, and needs further tuning.

## Method

LERF enables pixel-aligned queries of the distilled 3D CLIP embeddings without relying on region proposals, masks, or fine-tuning, supporting long-tail open-vocabulary queries hierarchically across the volume.

### Multi-scale supervision

To supervise language embeddings, we pre-compute an image pyramid of CLIP features for each training view. Then, each sampled ray during optimization is supervised by interpolating the CLIP embedding within this pyramid.

<img id="lerf_multiscale" src="https://www.lerf.io/data/clip_features.png" style="background-color:white;" width="100%">

### LERF Optimization

LERF optimizes a dense, multi-scale language 3D field by volume rendering CLIP embeddings along training rays, supervising these embeddings with multi-scale CLIP features across multi-view training images.

Inspired by Distilled Feature Fields (DFF), we use DINO features to regularize CLIP features. This leads to qualitative improvements in object boundaries, as CLIP embeddings in 3D can be sensitive to floaters and regions with few views.

After optimization, LERF can extract 3D relevancy maps for language queries interactively in real-time.

<img id="lerf_render" src="https://www.lerf.io/data/nerf_render.png" style="background-color:white;" width="100%">

### Visualizing relevancy for text queries

Set the "Output Render" type to `relevancy_0`, and enter the text query in the "LERF Positives" textbox (see image). The output render will show the 3D relevancy map for the query. View the [project page](https://lerf.io) for more examples and details about the relevancy map normalization.

<center>
<img id="lerf_viewer" src="https://www.lerf.io/data/lerf_screen.png" width="40%">
</center>

## Results

For results, view the [project page](https://lerf.io)!

Datasets used in the original paper can be found [here](https://drive.google.com/drive/folders/1LUzwEvBCE19PNYcwfmrG-9FLpZLbi4on?usp=sharing).

```none
@article{lerf2023,
 author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
 title = {LERF: Language Embedded Radiance Fields},
 journal = {arXiv preprint arXiv:2303.09553},
 year = {2023},
}
```
## nerfology/methods

### tetranerf.md

# Tetra-NeRF

<h4>Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra</h4>

```{button-link} https://jkulhanek.com/tetra-nerf
:color: primary
:outline:
Paper Website
```

```{button-link} https://github.com/jkulhanek/tetra-nerf
:color: primary
:outline:
Code
```

<video id="teaser" muted autoplay playsinline loop controls width="100%">
    <source id="mp4" src="https://jkulhanek.com/tetra-nerf/resources/intro-video.mp4" type="video/mp4">
</video>

**SfM input pointcloud is triangulated and resulting tetrahedra is used as the radiance field representation**

## Installation

First, make sure to install the following:
```
CUDA (>=11.3)
PyTorch (>=1.12.1)
Nerfstudio (>=0.2.0)
OptiX (>=7.2, preferably 7.6)
CGAL
CMake (>=3.22.1)
```
Follow the [installation section](https://github.com/jkulhanek/tetra-nerf/blob/master/README.md#installation) in the tetra-nerf repository

Finally, you can install **Tetra-NeRF** by running:
```bash
pip install git+https://github.com/jkulhanek/tetra-nerf
```

## Running Tetra-NeRF on custom data
Details for running Tetra-NeRF can be found [here](https://github.com/jkulhanek/tetra-nerf).

```bash
python -m tetranerf.scripts.process_images --path <data folder>
python -m tetranerf.scripts.triangulate --pointcloud <data folder>/sparse.ply --output <data folder>/sparse.th
ns-train tetra-nerf --pipeline.model.tetrahedra-path <data folder>/sparse.th minimal-parser --data <data folder>
```

Three following variants of Tetra-NeRF are provided:

| Method                | Description                            | Memory  | Quality |
| --------------------- | -------------------------------------- | ------- | ------- |
| `tetra-nerf-original` | Official implementation from the paper | ~18GB*  | Good    |
| `tetra-nerf`          | Different sampler - faster and better  | ~16GB*  | Best    |

*Depends on the size of the input pointcloud, estimate is given for a larger scene (1M points)

## Method
![method overview](https://jkulhanek.com/tetra-nerf/resources/overview-white.svg)<br>
The input to Tetra-NeRF is a point cloud which is triangulated to get a set of tetrahedra used to represent the radiance field. Rays are sampled, and the field is queried. The barycentric interpolation is used to interpolate tetrahedra vertices, and the resulting features are passed to a shallow MLP to get the density and colours for volumetric rendering.<br>

[![demo blender lego (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/blender-lego-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=blender-lego-sparse)
[![demo mipnerf360 garden (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/360-garden-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-garden-sparse)
[![demo mipnerf360 garden (sparse)](https://jkulhanek.com/tetra-nerf/resources/images/360-bonsai-sparse-100k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-bonsai-sparse)
[![demo mipnerf360 kitchen (dense)](https://jkulhanek.com/tetra-nerf/resources/images/360-kitchen-dense-300k-animated-cover.gif)](https://jkulhanek.com/tetra-nerf/demo.html?scene=360-kitchen-dense)


## Existing checkpoints and predictions
For an easier comparisons with Tetra-NeRF, published checkpoints and predictions can be used:

| Dataset  | Checkpoints | Predictions | Input tetrahedra |
| -------- | ----------- | ----------- | ---------------- |
| Mip-NeRF 360 (public scenes) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360-public-tetrahedra.tar.gz) |
| Blender | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/blender-tetrahedra.tar.gz) |
| Tanks and Temples | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-checkpoints.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-predictions.tar.gz) | [download](https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/nsvf-tanks-and-temples-tetrahedra.tar.gz) |

## developer_guides

### config.md

# Customizable configs

Our dataclass configs allow you to easily plug in different permutations of models, dataloaders, modules, etc.
and modify all parameters from a typed CLI supported by [tyro](https://pypi.org/project/tyro/).

### Base components

All basic, reusable config components can be found in `nerfstudio/configs/base_config.py`. The `Config` class at the bottom of the file is the upper-most config level and stores all of the sub-configs needed to get started with training.

You can browse this file and read the attribute annotations to see what configs are available and what each specifies.

### Creating new configs

If you are interested in creating a brand new model or data format, you will need to create a corresponding config with associated parameters you want to expose as configurable.

Let's say you want to create a new model called Nerfacto. You can create a new `Model` class that extends the base class as described [here](pipelines/models.ipynb). Before the model definition, you define the actual `NerfactoModelConfig` which points to the `NerfactoModel` class (make sure to wrap the `_target` classes in a `field` as shown below).

:::{admonition} Tip
:class: info

You can then enable type/auto complete on the config passed into the `NerfactoModel` by specifying the config type below the class definition.
:::

```python
"""nerfstudio/models/nerfacto.py"""

@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    ...

class NerfactoModel(Model):
     """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig
    ...
```

The same logic applies to all other custom configs you want to create. For more examples, you can see `nerfstudio/data/dataparsers/nerfstudio_dataparsers.py`, `nerfstudio/data/datamanagers.py`.

:::{admonition} See Also
:class: seealso

For how to create the actual data and model classes that follow the configs, please refer to [pipeline overview](pipelines/index.rst).
:::

### Updating method configs

If you are interested in creating a new model config, you will have to modify the `nerfstudio/configs/method_configs.py` file. This is where all of the configs for implemented models are housed. You can browse this file to see how we construct various existing models by modifying the `Config` class and specifying new or modified default components.

For instance, say we created a brand new model called Nerfacto that has an associated `NerfactoModelConfig`, we can specify the following new Config by overriding the pipeline and optimizers attributes appropriately.

```python
"""nerfstudio/configs/method_configs.py"""

method_configs["nerfacto"] = Config(
    method_name="nerfacto",
    pipeline=VanillaPipelineConfig(
        model=NerfactoModelConfig(eval_num_rays_per_chunk=1 << 14),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
)
```

After placing your new `Config` class into the `method_configs` dictionary, you can provide a description for the model by updating the `descriptions` dictionary at the top of the file.

### Modifying from CLI

Often times, you just want to play with the parameters of an existing model without having to specify a new one. You can easily do so via CLI. Below, we showcase some useful CLI commands.

- List out all existing models

  ```bash
  ns-train --help
  ```

- List out all existing configurable parameters for `{METHOD_NAME}`

  ```bash
  ns-train {METHOD_NAME} --help
  ```

- Change the train/eval dataset

  ```bash
  ns-train {METHOD_NAME} --data DATA_PATH
  ```

- Enable the viewer

  ```bash
  ns-train {METHOD_NAME} --vis viewer
  ```

- See what options are available for the specified dataparser (e.g. blender-data)

  ```bash
  ns-train {METHOD_NAME} {DATA_PARSER} --help
  ```

- Run with changed dataparser attributes and viewer on
  ```bash
  # NOTE: the dataparser and associated configurations go at the end of the command
  ns-train {METHOD_NAME} --vis viewer {DATA_PARSER} --scale-factor 0.5
  ```
## developer_guides

### new_methods.md

# Adding a New Method

Nerfstudio aims to offer researchers a codebase that they can utilize to extend and develop novel methods. Our vision is for users to establish a distinct repository that imports nerfstudio and overrides pipeline components to cater to specific functionality requirements of the new approach. If any of the new features require modifications to the core nerfstudio repository and can be generally useful, we encourage you to submit a PR to enable others to benefit from it.

You can use the [nerfstudio-method-template](https://github.com/nerfstudio-project/nerfstudio-method-template) repository as a minimal guide to register your new methods. Examples are often the best way to learn, take a look at the [LERF](https://github.com/kerrj/lerf) repository for a good example of how to extend and use nerfstudio in your projects.

## File Structure

We recommend the following file structure:

```
├── my_method
│   ├── __init__.py
│   ├── my_config.py
│   ├── custom_pipeline.py [optional]
│   ├── custom_model.py [optional]
│   ├── custom_field.py [optional]
│   ├── custom_datamanger.py [optional]
│   ├── custom_dataparser.py [optional]
│   ├── ...
├── pyproject.toml
```

## Registering custom model with nerfstudio

In order to extend the Nerfstudio and register your own methods, you can package your code as a python package
and register it with Nerfstudio as a `nerfstudio.method_configs` entrypoint in the `pyproject.toml` file.
Nerfstudio will automatically look for all registered methods and will register them to be used
by methods such as `ns-train`.

First create a config file:

```python
"""my_method/my_config.py"""

from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

MyMethod = MethodSpecification(
  config=TrainerConfig(
    method_name="my-method",
    pipeline=...
    ...
  ),
  description="Custom description"
)
```

Then create a `pyproject.toml` file. This is where the entrypoint to your method is set and also where you can specify additional dependencies required by your codebase.

```python
"""pyproject.toml"""

[project]
name = "my_method"

dependencies = [
    "nerfstudio" # you may want to consider pinning the version, ie "nerfstudio==0.1.19"
]

[tool.setuptools.packages.find]
include = ["my_method*"]

[project.entry-points.'nerfstudio.method_configs']
my-method = 'my_method.my_config:MyMethod'
```

finally run the following to register the method,

```
pip install -e .
```

When developing a new method you don't always want to install your code as a package.
Instead, you may use the `NERFSTUDIO_METHOD_CONFIGS` environment variable to temporarily register your custom method.

```
export NERFSTUDIO_METHOD_CONFIGS="my-method=my_method.my_config:MyMethod"
```

The `NERFSTUDIO_METHOD_CONFIGS` environment variable additionally accepts a function or derived class to temporarily register your custom method.

```python
"""my_method/my_config.py"""

from dataclasses import dataclass
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

def MyMethodFunc():
    return MethodSpecification(
      config=TrainerConfig(...)
      description="Custom description"
    )

@dataclass
class MyMethodClass(MethodSpecification):
    config: TrainerConfig = TrainerConfig(...)
    description: str = "Custom description"
```

## Registering custom dataparser with nerfstudio

We also support adding new dataparsers in a similar way. In order to extend the NeRFstudio and register a customized dataparser, you can register it with Nerfstudio as a `nerfstudio.dataparser_configs` entrypoint in the `pyproject.toml` file. Nerfstudio will automatically look for all registered dataparsers and will register them to be used by methods such as `ns-train`.

You can declare the dataparser in the same config file:

```python
"""my_method/my_config.py"""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from my_method.custom_dataparser import CustomDataparserConfig

MyDataparser = DataParserSpecification(config=CustomDataparserConfig())
```

Then add the following lines in the `pyproject.toml` file, where the entrypoint to the new dataparser is set.

```python
"""pyproject.toml"""

[project]
name = "my_method"

[project.entry-points.'nerfstudio.dataparser_configs']
custom-dataparser = 'my_method.my_config:MyDataparser'
```

finally run the following to register the dataparser.

```
pip install -e .
```

Similarly to the method development, you can also use environment variables to register dataparsers.
Use the `NERFSTUDIO_DATAPARSER_CONFIGS` environment variable:

```
export NERFSTUDIO_DATAPARSER_CONFIGS="my-dataparser=my_package.my_config:MyDataParser"
```

Same as with custom methods, `NERFSTUDIO_DATAPARSER_CONFIGS` environment variable additionally accepts a function or derived class to temporarily register your custom method.

## Running custom method

After registering your method you should be able to run the method with,

```
ns-train my-method --data DATA_DIR
```

## Adding to the _nerf.studio_ documentation

We invite researchers to contribute their own methods to our online documentation. You can find more information on how to do this {ref}`here<own_method_docs>`.
## developer_guides/pipelines

### pipelines.md

# Pipelines

```{image} imgs/pipeline_pipeline-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_pipeline-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a Pipeline?

The Pipeline contains all the code you need to implement a NeRF method. There are two main functions that you need to implement for the Pipeline.

```python
class Pipeline(nn.Module):

    datamanager: DataManager
    model: Model

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

    @profiler.time_function
    def get_eval_loss_dict(self, step: int):
        """This function gets your evaluation loss dict. It needs to get the data
        from the DataManager and feed it to the model's forward function

        Args:
            step: current iteration step
        """
```

## Vanilla Implementation

Here you can see a simple implementation of the get_train_loss_dict from the VanillaPipeline. Essentially, all the pipeline has to do is route data from the DataManager to the Model.

```python
@profiler.time_function
def get_train_loss_dict(self, step: int):
    ray_bundle, batch = self.datamanager.next_train(step)
    model_outputs = self.model(ray_bundle)
    metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
    loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
    return model_outputs, loss_dict, metrics_dict
```

## Creating Custom Methods

:::{admonition} Note
:class: info

The VanillaPipeline works for most of our methods.
:::

We also have a DynamicBatchPipeline that is used with InstantNGP to dynamically choose the number of rays to use per training and evaluation iteration.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/pipelines/dynamic_batch.py
:color: primary
:outline:
See the code!
```## developer_guides/pipelines

### fields.md

# Fields

```{image} imgs/pipeline_field-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_field-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a Field?

A Field is a model component that associates a region of space with some sort of quantity. In the most typical case, the input to a field is a 3D location and viewing direction, and the output is density and color. Let's take a look at the code.

```python
class Field(nn.Module):
    """Base class for fields."""

    @abstractmethod
    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType[..., 1], TensorType[..., "num_features"]]:
        """Computes and returns the densities. Returns a tensor of densities and a tensor of features.

        Args:
            ray_samples: Samples locations to compute density.
        """

    @abstractmethod
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        """Computes and returns the colors. Returns output field values.

        Args:
            ray_samples: Samples locations to compute outputs.
            density_embedding: Density embeddings to condition on.
        """

    def forward(self, ray_samples: RaySamples):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        density, density_embedding = self.get_density(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)

        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore
        return field_outputs
```

## Separate density and outputs

The forward function is the main function you'll use, which takes in RaySamples returns quantities for each sample. You'll notice that the get_density function is called for every field, followed by the get_outputs function.

The get_outputs function is what you need to implement to return custom data. For example, check out of SemanticNerfField where we rely on different FieldHeads to produce correct dimensional outputs for typical quantities. Our implemented FieldHeads have the following FieldHeadNames names.

```python
class FieldHeadNames(Enum):
    """Possible field outputs"""

    RGB = "rgb"
    SH = "sh"
    DENSITY = "density"
    UNCERTAINTY = "uncertainty"
    TRANSIENT_RGB = "transient_rgb"
    TRANSIENT_DENSITY = "transient_density"
    SEMANTICS = "semantics"
```

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/field_components/field_heads.py
:color: primary
:outline:
See the code!
```

Sometimes all you need is the density from a Field, so we have a helper method called density_fn which takes positions and returns densities.

## Using Frustums instead of positions

Let's say you want to query a region of space, rather than a point. Our RaySamples data structure contains Frustums which can be used for exactly this purpose. This enables methods like Mip-NeRF to be implemented in our framework.

```python
@dataclass
class RaySamples(TensorDataclass):
    """Samples along a ray"""

    frustums: Frustums
    """Frustums along ray."""
    ...

@dataclass
class Frustums(TensorDataclass):
    """Describes region of space as a frustum."""

    origins: TensorType["bs":..., 3]
    """xyz coordinate for ray origin."""
    directions: TensorType["bs":..., 3]
    """Direction of ray."""
    starts: TensorType["bs":..., 1]
    """Where the frustum starts along a ray."""
    ends: TensorType["bs":..., 1]
    """Where the frustum ends along a ray."""
    pixel_area: TensorType["bs":..., 1]
    """Projected area of pixel a distance 1 away from origin."""
    ...
```

Take a look at our RaySamples class for more information on the input to our Field classes.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/fields/base_field.py
:color: primary
:outline:
See the code!
```
## developer_guides/pipelines

### dataparsers.md

# DataParsers

```{image} imgs/pipeline_parser-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_parser-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a DataParser?

The dataparser returns `DataparserOutputs`, which puts all the various datasets into a common format. The DataparserOutputs should be lightweight, containing filenames or other meta information which can later be processed by actual PyTorch Datasets and Dataloaders. The common format makes it easy to add another DataParser. All you have to do is implement the private method `_generate_dataparser_outputs` shown below.

```python
@dataclass
class DataparserOutputs:
    """Dataparser outputs for the which will be used by the DataManager
    for creating RayBundle and RayGT objects."""

    image_filenames: List[Path]
    """Filenames for the images."""
    cameras: Cameras
    """Camera object storing collection of camera information in dataset."""
    alpha_color: Optional[TensorType[3]] = None
    """Color of dataset background."""
    scene_box: SceneBox = SceneBox()
    """Scene box of dataset. Used to bound the scene or provide the scene scale depending on model."""
    mask_filenames: Optional[List[Path]] = None
    """Filenames for any masks that are required"""
    metadata: Dict[str, Any] = to_immutable_dict({})
    """Dictionary of any metadata that be required for the given experiment.
    Will be processed by the InputDataset to create any additional tensors that may be required.
    """
    dataparser_transform: TensorType[3, 4] = torch.eye(4)[:3, :]
    """Transform applied by the dataparser."""
    dataparser_scale: float = 1.0
    """Scale applied by the dataparser."""

@dataclass
class DataParser:

    @abstractmethod
    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        """Abstract method that returns the dataparser outputs for the given split.

        Args:
            split: Which dataset split to generate (train/test).

        Returns:
            DataparserOutputs containing data for the specified dataset and split
        """
```

## Example

Here is an example where we implement a DataParser for our Nerfstudio data format.

```python
@dataclass
class NerfstudioDataParserConfig(DataParserConfig):
    """Nerfstudio dataset config"""

    _target: Type = field(default_factory=lambda: Nerfstudio)
    """target class to instantiate"""
    data: Path = Path("data/nerfstudio/poster")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    downscale_factor: Optional[int] = None
    """How much to downscale images. If not set, images are chosen such that the max dimension is <1600px."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "poses"
    """The method to use to center the poses."""
    auto_scale_poses: bool = True
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    train_split_fraction: float = 0.9
    """The fraction of images to use for training. The remaining images are for eval."""
    depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

@dataclass
class Nerfstudio(DataParser):
    """Nerfstudio DatasetParser"""

    config: NerfstudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        meta = load_from_json(self.config.data / "transforms.json")
        image_filenames = []
        poses = []
        ...
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
        )
        return dataparser_outputs
```

## Train and Eval Logic

The DataParser will generate a train and eval DataparserOutputs depending on the `split` argument. For example, here is how you'd initialize some `InputDataset` classes that live in the DataManager. Because our DataparserOutputs maintain a common form, our Datasets should be plug-and-play. These datasets will load images needed to supervise the model with `RayGT` objects.

```python
config = NerfstudioDataParserConfig()
dataparser = config.setup()
# train dataparser
dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
input_dataset = InputDataset(dataparser_outputs)
```

You can also pull out information from the DataParserOutputs for other DataManager components, such as the RayGenerator. The RayGenerator generates RayBundle objects from camera and pixel indices.

```python
ray_generator = RayGenerator(dataparser_outputs.cameras)
```

## Included DataParsers

```{toctree}
---
maxdepth: 2
---

../../reference/api/data/dataparsers
```
## developer_guides/pipelines

### index.rst

```rst
Pipelines overview
-------------------------

Here we describe what a Pipeline is and how it works. You can see an overview figure with the major Pipeline components below.

.. image:: imgs/pipeline_overview-light.png
  :width: 600
  :align: center
  :alt: pipeline figure
  :class: only-light

.. image:: imgs/pipeline_overview-dark.png
  :width: 600
  :align: center
  :alt: pipeline figure
  :class: only-dark


.. admonition:: Note

  RayGT and RayOutputs are currently dictionaries. In the future, they will be typed objects.


Why Pipelines?
==========================

Our goal is for any NeRF paper to be implemented as a Pipeline.

The Pipeline is composed of two major components, namely the DataManager and the Model. The DataManager is responsible for loading data and generating RayBundle and RayGT objects. RayBundles are the input to the forward pass of the Model. These are needed for both training and inference time. RayGT objects, however, are needed only during training to calculate the losses in the Loss Dict.

RayBundle objects describe origins and viewing directions. The model will take these rays and render them into quantities as RayOutputs. RayGT contains the necessary ground truth (GT) information needed to compute losses. For example, the GT pixel values can be used to supervise the rendered rays with an L2 loss.

In the following sections, we describe the Pipeline components and look at their code.

.. toctree::
    :maxdepth: 1

    dataparsers
    datamanagers
    models
    fields
    pipelines

Implementing NeRF Papers
==========================

Let's say you want to create a custom Pipeline that has a custom DataManager and a custom Model. Perhaps you care about dynamically adding cameras to the DataManager during training or you want to importance sample and generate rays from pixels where the loss is high. This can be accomplished by mixing and matching components into a Pipeline. The following guide will take you through an example of this.

This guide is coming soon!

```

## developer_guides/pipelines

### datamanagers.md

# DataManagers

```{image} imgs/pipeline_datamanager-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_datamanager-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a DataManager?

The DataManager returns RayBundle and RayGT objects. Let's first take a look at the most important abstract methods required by the DataManager.

```python
class DataManager(nn.Module):
    """Generic data manager's abstract class
    """

    @abstractmethod
    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data for train."""

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data for eval."""

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        """Returns the next eval image.

        Returns:
            The image index from the eval dataset, the CameraRayBundle, and the RayGT dictionary.
        """
```

## Example

We've implemented a VanillaDataManager that implements the standard logic of most NeRF papers. It will randomly sample training rays with corresponding ground truth information, in RayBundle and RayGT objects respectively. The config for the VanillaDataManager is the following.

```python
@dataclass
class VanillaDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    """target class to instantiate"""
    dataparser: AnnotatedDataParserUnion = BlenderDataParserConfig()
    """specifies the dataparser used to unpack the data"""
    train_num_rays_per_batch: int = 1024
    """number of rays per batch to use per training iteration"""
    train_num_images_to_sample_from: int = -1
    """number of images to sample during training iteration"""
    eval_num_rays_per_batch: int = 1024
    """number of rays per batch to use per eval iteration"""
    eval_num_images_to_sample_from: int = -1
    """number of images to sample during eval iteration"""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig()
    """specifies the camera pose optimizer used during training"""
```

Let's take a quick look at how the `next_train` method is implemented. Here we sample images, then pixels, and then return the RayBundle and RayGT information.

```python
def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
    """Returns the next batch of data from the train dataloader."""
    self.train_count += 1
    # sample a batch of images
    image_batch = next(self.iter_train_image_dataloader)
    # sample pixels from this batch of images
    batch = self.train_pixel_sampler.sample(image_batch)
    ray_indices = batch["indices"]
    # generate rays from this image and pixel indices
    ray_bundle = self.train_ray_generator(ray_indices)
    # return RayBundle and RayGT information
    return ray_bundle, batch
```

You can see our code for more details.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/datamanagers/base_datamanager.py
:color: primary
:outline:
See the code!
```

## Creating Your Own

We currently don't have other implementations because most papers follow the VanillaDataManager implementation. However, it should be straightforward to add a VanillaDataManager with logic that progressively adds cameras, for instance, by relying on the step and modifying RayBundle and RayGT generation logic.
## developer_guides/pipelines

### models.md

# Models

```{image} imgs/pipeline_model-light.png
:align: center
:class: only-light
:width: 600
```

```{image} imgs/pipeline_model-dark.png
:align: center
:class: only-dark
:width: 600
```

## What is a Model?

A Model is probably what you think of when you think of a NeRF paper. Often the phrases "Model" and "Method" are used interchangeably and for this reason, our implemented [Methods](/nerfology/methods/index) typically only change the model code.

A model, at a high level, takes in regions of space described by RayBundle objects, samples points along these rays, and returns rendered values for each ray. So, let's take a look at what it takes to create your own model!

## Functions to Implement

[The code](https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/base_model.py) is quite verbose, so here we distill the most important functions with succint descriptions.

```python
class Model:

    config: ModelConfig
    """Set the model config so that Python gives you typed autocomplete!"""

    def populate_modules(self):
        """Set the fields and modules."""

        # Fields

        # Ray Samplers

        # Colliders

        # Renderers

        # Losses

        # Metrics

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Returns the parameter groups needed to optimizer your model components."""

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
        ) -> List[TrainingCallback]:
        """Returns the training callbacks, such as updating a density grid for Instant NGP."""

    def get_outputs(self, ray_bundle: RayBundle):
        """Process a RayBundle object and return RayOutputs describing quanties for each ray."""

    def get_metrics_dict(self, outputs, batch):
        """Returns metrics dictionary which will be plotted with comet, wandb or tensorboard."""

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        """Returns a dictionary of losses to be summed which will be your loss."""

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Returns a dictionary of images and metrics to plot. Here you can apply your colormaps."""
```

## Pythonic Configs with Models

Our config system is most useful when it comes to models. Let's take a look at our Nerfacto model config.

```python
@dataclass
class NerfactoModelConfig(ModelConfig):
    """Nerfacto Model Config"""

    _target: Type = field(default_factory=lambda: NerfactoModel)
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    background_color: Literal["background", "last_sample"] = "last_sample"
    """Whether to randomize the background color."""
    num_proposal_samples_per_ray: Tuple[int] = (64,)
    """Number of samples per ray for the proposal network."""
    num_nerf_samples_per_ray: int = 64
    """Number of samples per ray for the nerf network."""
    num_proposal_network_iterations: int = 1
    """Number of proposal network iterations."""
    use_same_proposal_network: bool = False
    """Use the same proposal network. Otherwise use different ones."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""
    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing."""
    use_average_appearance_embedding: bool = True
    """Whether to use average appearance embedding or zeros for inference."""
    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights."""
    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function."""
    use_single_jitter: bool = True
    """Whether use single jitter or not for the proposal networks."""
```

There are a lot of options! Thankfully, our config system makes this easy to handle. If you want to add another argument, you simply add a value to this config and when you type in `ns-train nerfacto --help`, it will show in the terminal as a value you can modify.

Furthermore, you have Python autocomplete and static checking working in your favor. At the top of every Model, we specify the config and then can easily pull of values throughout the implementation. Let's take a look at the beginning of the NerfactoModel implementation.

```python
class NerfactoModel(Model):
    """Nerfacto model

    Args:
        config: Nerfacto configuration to instantiate model
    """

    config: NerfactoModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        ...
        # Fields
        self.field = TCNNNerfactoField(
            self.scene_box.aabb,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding, # notice self.config
        )
        ...
        # Renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color) # notice self.config
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()
```

We invite you to take a look at the Nerfacto model and others to see how our models are formatted.

```{button-link} https://github.com/nerfstudio-project/nerfstudio/blob/master/nerfstudio/models/nerfacto.py
:color: primary
:outline:
See the code!
```

## Implementing a Model

Now that you understand how the model is structured, you can create a model by populating these functions. We provide a library of model components to pull from when creating your model. Check out those tutorials here!

One of these components is a Field, which you can learn more about in the next section. Fields associate a quantity of space with a value (e.g., density and color) and are used in every model.
## developer_guides/viewer

### local_viewer.md

# (Legacy Viewer) Local Server

**Note:** this doc only applies to the legacy version of the viewer, which was the default in in Nerfstudio versions `<=0.3.4`. It was deprecated starting Nerfstudio version `1.0.0`, where it needs to be opted into via the `--vis viewer_legacy` argument.

---

If you are unable to connect to `https://viewer.nerf.studio`, want to use Safari, or want to develop the viewer codebase, you can launch your own local viewer.

## Installing Dependencies

```shell
cd nerfstudio/viewer/app
```

Install npm (to install yarn) and yarn

```shell
sudo apt-get install npm
npm install --global yarn
```

Install nvm and set the node version
Install nvm with [instructions](https://heynode.com/tutorial/install-nodejs-locally-nvm/).

```shell
nvm install 17.8.0
```

Now running `node --version` in the shell should print "v17.8.0".
Install package.json dependencies and start the client viewer app:

```shell
yarn install
```

## Launch the web client

From the `nerfstudio/viewer/app` folder, run:

```shell
yarn start
```

The local webserver runs on port 4000 by default,
so when `ns-train` is running, you can connect to the viewer locally at
[http://localhost:4000/?websocket_url=ws://localhost:7007](http://localhost:4000/?websocket_url=ws://localhost:7007)

## FAQ

### Engine node incompatible

While running `yarn install`, you run into: `The engine "node" is incompatible with this module.`

**Solution**:

Install nvm with instructions at [instructions](https://heynode.com/tutorial/install-nodejs-locally-nvm/).

```shell
nvm install 17.8.0
```

If you cannot install nvm, try ignoring the engines

```shell
yarn install --ignore-engines
```
## developer_guides/viewer

### index.md

# Viewer

> We have a real-time web viewer that requires no installation. It's available at [https://viewer.nerf.studio/](https://viewer.nerf.studio/), where you can connect to your training job.

The viewer is built on [Viser](https://github.com/brentyi/viser/tree/main/viser) using [ThreeJS](https://threejs.org/) and packaged into a [ReactJS](https://reactjs.org/) application. This client viewer application will connect via a websocket to a server running on your machine.

```{toctree}
:titlesonly:

custom_gui
viewer_control
local_viewer
```

## Acknowledgements and references

We thank the authors and contributors to the following repos, which we've started, used, and modified for our use-cases.

- [Viser](https://github.com/brentyi/viser/) - made by [Brent Yi](https://github.com/brentyi)
- [meshcat-python](https://github.com/rdeits/meshcat-python) - made by [Robin Deits](https://github.com/rdeits)
- [meshcat](https://github.com/rdeits/meshcat) - made by [Robin Deits](https://github.com/rdeits)
- [ThreeJS](https://threejs.org/)
- [ReactJS](https://reactjs.org/)
## developer_guides/viewer

### custom_gui.md

# Custom GUI

We provide support for custom viewer GUI elements that can be defined in any `nn.Module`. Although we don't have any specific use cases in mind, here are some examples of what can be achieved with this feature:

- Using text input to modify the rendering
- Logging numerical values to the viewer
- Using checkboxes to turn off and on losses
- Using a dropdown to switch between appearances

## Adding an Element

To define a custom element, create an instance of one of the provided classes in `nerfstudio.viewer.viewer_elements`, and assign it as a class variable in your `nn.Module`.

```python
from nerfstudio.viewer.viewer_elements import ViewerNumber

class MyClass(nn.Module):#must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.custom_value = ViewerNumber(name="My Value", default_value=1.0)
```
**Element Hierarchy**
The viewer recursively searches all `nn.Module` children of the base `Pipeline` object, and arranges parameters into folders based on their variable names.
For example, a `ViewerElement` defined in `pipeline.model.field` will be in the "Custom/model/field" folder in the GUI.

**Reading the value**
To read the value of a custom element, simply access its `value` attribute. In this case it will be `1.0` unless modified by the user in the viewer.

```python
current_value = self.custom_value.value
```

**Callbacks**
You can register a callback that will be called whenever a new value for your GUI element is available. For example, one can use a callback to update config parameters when elements are changed:
```python
def on_change_callback(handle: ViewerCheckbox) -> None:
    self.config.example_parameter = handle.value

self.custom_checkbox = ViewerCheckbox(
    name="Checkbox",
    default_value=False,
    cb_hook=on_change_callback,
)
```

**Thread safety**
Note that `ViewerElement` values can change asynchronously to model execution. So, it's best practice to store the value of a viewer element once at the beginning
of a forward pass and refer to the static variable afterwards.
```python
class MyModel(Model):
    def __init__(self):
        self.slider = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)

    def get_outputs(self,ray)
        slider_val = self.slider.value
        #self.slider.value could change after this, unsafe to use

```


**Writing to the element**
You can write to a viewer element in Python, which provides a convenient way to track values in your code without the need for comet/wandb/tensorboard or relying on `print` statements.

```python
self.custom_value.value = x
```

:::{admonition} Warning
:class: warning

Updating module state while training can have unexpected side effects. It is up to the user to ensure that GUI actions are safe. Conditioning on `self.training` can help determine whether effects are applied during forward passes for training or rendering.
:::

## Example Elements

```{image} imgs/custom_controls.png
:align: center
:width: 400
```

This was created with the following elements:

```python
from nerfstudio.viewer.viewer_elements import *

class MyModel(Model):
    def __init__(self):
        self.a = ViewerButton(name="My Button", cb_hook=self.handle_btn)
        self.b = ViewerNumber(name="Number", default_value=1.0)
        self.c = ViewerCheckbox(name="Checkbox", default_value=False)
        self.d = ViewerDropdown(name="Dropdown", default_value="A", options=["A", "B"])
        self.e = ViewerSlider(name="Slider", default_value=0.5, min_value=0.0, max_value=1.0)
        self.f = ViewerText(name="Text", default_value="Hello World")
        self.g = ViewerVec3(name="3D Vector", default_value=(0.1, 0.7, 0.1))

        self.rgb_renderer = RGBRenderer()
...
class RGBRenderer(nn.Module):
    def __init__(self):
        #lives in "Custom/model/rgb_renderer" GUI folder
        self.a = ViewerRGB(name="F", default_value=(0.1, 0.7, 0.1))
...
```

For more information on the available classes and their arguments, refer to the [API documentation](../../reference/api/viewer.rst)
## developer_guides/viewer

### viewer_control.md

# Python Viewer Control

Similar to [`ViewerElements`](./custom_gui.md), Nerfstudio includes supports a Python interface to the viewer through which you can:

* Set viewer camera pose and FOV
* Set viewer scene crop
* Retrieve the current viewer camera matrix
* Install listeners for click events inside the viewer window

## Usage

First, instantiate a `ViewerControl` object as a class variable inside a model file.
Just like `ViewerElements`, you can create an instance inside any class which inherits from `nn.Module`
and is contained within the `Pipeline` object (for example the `Model`)

```python
from nerfstudio.viewer.viewer_elements import ViewerControl

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
```
## Get Camera Matrix
To get the current camera intrinsics and extrinsics, use the `get_camera` function. This returns a `nerfstudio.cameras.cameras.Cameras` object. This object can be used to generate `RayBundles`, retrieve intrinsics and extrinsics, and more.

```python
from nerfstudio.viewer.viewer_elements import ViewerControl, ViewerButton

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        ...
        def button_cb(button):
            # example of using the get_camera function, pass img width and height
            # returns a Cameras object with 1 camera
            camera = self.viewer_control.get_camera(100,100)
            if camera is None:
                # returns None when the viewer is not connected yet
                return
            # get the camera pose
            camera_extrinsics_matrix = camera.camera_to_worlds[0,...]  # 3x4 matrix
            # generate image RayBundle
            bundle = camera.generate_rays(camera_indices=0)
            # Compute depth, move camera, or whatever you want
            ...
        self.viewer_button = ViewerButton(name="Dummy Button",cb_hook=button_cb)
```

## Set Camera Properties
You can set the viewer camera position and FOV from python.
To set position, you must define a new camera position as well as a 3D "look at" point which the camera aims towards.
```python
from nerfstudio.viewer.viewer_elements import ViewerControl,ViewerButton

class MyModel(nn.Module):  # Must inherit from nn.Module
    def __init__(self):
        ...
        def aim_at_origin(button):
            # instant=False means the camera smoothly animates
            # instant=True means the camera jumps instantly to the pose
            self.viewer_control.set_pose(position=(1,1,1),look_at=(0,0,0),instant=False)
        self.viewer_button = ViewerButton(name="Dummy Button",cb_hook=button_cb)
```

## Scene Click Callbacks
We forward *single* clicks inside the viewer to the ViewerControl object, which you can use to interact with the scene. To do this, register a callback using `register_click_cb()`. The click is defined to be a ray that starts at the camera origin and passes through the click point on the screen, in world coordinates. 

```python
from nerfstudio.viewer.viewer_elements import ViewerControl,ViewerClick

class MyModel(nn.Module):  # must inherit from nn.Module
    def __init__(self):
        # Must be a class variable
        self.viewer_control = ViewerControl()  # no arguments
        def click_cb(click: ViewerClick):
            print(f"Click at {click.origin} in direction {click.direction}")
        self.viewer_control.register_click_cb(click_cb)
```

You can also use `unregister_click_cb()` to remove callbacks that are no longer needed. A good example is a "Click on Scene" button, that when pressed, would register a callback that would wait for the next click, and then unregister itself.
```python
    ...
    def button_cb(button: ViewerButton):
        def click_cb(click: ViewerClick):
            print(f"Click at {click.origin} in direction {click.direction}")
            self.viewer_control.unregister_click_cb(click_cb)
        self.viewer_control.register_click_cb(click_cb)
```

### Thread safety
Just like `ViewerElement` callbacks, click callbacks are asynchronous to training and can potentially interrupt a call to `get_outputs()`.
## developer_guides/debugging_tools

### benchmarking.md

# Benchmarking workflow

We make it easy to benchmark your new NeRF against the standard Blender dataset.

## Launching training on Blender dataset

To start, you will need to train your NeRF on each of the blender objects.
To launch training jobs automatically on each of these items, you can call:

```bash

./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m {METHOD_NAME} [-s] [-v {VIS}] [{GPU_LIST}]
```

Simply replace the arguments in brackets with the correct arguments.

- `-m {METHOD_NAME}`: Name of the method you want to benchmark (e.g. `nerfacto`, `mipnerf`).
- `-s`: Launch a single job per GPU.
- `-v {VIS}`: Use another visualization than wandb, which is the default. Other options are comet & tensorboard.
- `{GPU_LIST}`: (optional) Specify the list of gpus you want to use on your machine space separated. for instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute training jobs on the available GPUs.

:::{admonition} Tip
:class: info

To view all the arguments and annotations, you can run `./nerfstudio/scripts/benchmarking/launch_train_blender.sh --help`
:::

A full example would be:

- Specifying gpus

  ```bash
  ./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m nerfacto 0 1 2 3
  ```

- Automatically find available gpus
  ```bash
  ./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m nerfacto
  ```

The script will automatically launch training on all of the items and save the checkpoints in an output directory with the experiment name and current timestamp.

## Evaluating trained Blender models

Once you have launched training, and training converges, you can test your method with `nerfstudio/scripts/benchmarking/launch_eval_blender.sh`.

Say we ran a benchmark on 08-10-2022 for `instant-ngp`. By default, the train script will save the benchmarks in the following format:

```
outputs
└───blender_chair_2022-08-10
|   └───instant-ngp
|       └───2022-08-10_172517
|           └───config.yml
|               ...
└───blender_drums_2022-08-10
|   └───instant-ngp
|       └───2022-08-10_172517
|           └───config.yml
|               ...
...
```

If we wanted to run the benchmark on all the blender data for the above example, we would run:

```bash

./nerfstudio/scripts/benchmarking/launch_eval_blender.sh -m instant-ngp -o outputs/ -t 2022-08-10_172517 [{GPU_LIST}]
```

The flags used in the benchmarking script are defined as follows:

- `-m`: config name (e.g. `instant-ngp`). This should be the same as what was passed in for -c in the train script.
- `-o`: base output directory for where all of the benchmarks are stored (e.g. `outputs/`). Corresponds to the `--output-dir` in the base `Config` for training.
- `-t`: timestamp of benchmark; also the identifier (e.g. `2022-08-10_172517`).
- `-s`: Launch a single job per GPU.
- `{GPU_LIST}`: (optional) Specify the list of gpus you want to use on your machine space separated. For instance, if you want to use GPU's 0-3, you will need to pass in `0 1 2 3`. If left empty, the script will automatically find available GPU's and distribute evaluation jobs on the available GPUs.

The script will simultaneously run the benchmarking across all the objects in the blender dataset and calculates the PSNR/FPS/other stats. The results are saved as .json files in the `-o` directory with the following format:

```
outputs
└───instant-ngp
|   └───blender_chair_2022-08-10_172517.json
|   |   blender_ficus_2022-08-10_172517.json
|   |   ...
```

:::{admonition} Warning
:class: warning

Since we are running multiple backgrounded processes concurrently with this script, please note the terminal logs may be messy.
:::
## developer_guides/debugging_tools

### profiling.md

# Code profiling support

We provide built-in performance profiling capabilities to make it easier for you to debug and assess the performance of your code.

#### In-house profiler

You can use our built-in profiler. By default, it is enabled and will print at the termination of the program. You can disable it via CLI using the flag `--logging.no-enable-profiler`.


The profiler computes the average total time of execution for any function with the `@profiler.time_function` decorator.
For instance, if you wanted to profile the total time it takes to generate rays given pixel and camera indices via the `RayGenerator` class, you might want to time its `forward()` function. In that case, you would need to add the decorator to the function.

```python
"""nerfstudio/model_components/ray_generators.py""""

class RayGenerator(nn.Module):

    ...

    @profiler.time_function     # <-- add the profiler decorator before the function
    def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
        # implementation here
        ...
```

Alternatively, you can also time parts of the code:
```python
...

def forward(self, ray_indices: TensorType["num_rays", 3]) -> RayBundle:
    # implementation here
    with profiler.time_function("code1"):
      # some code here
      ...

    with profiler.time_function("code2"):
      # some code here
      ...
    ...
```


At termination of training or end of the training run, the profiler will print out the average execution time for all of the functions or code blocks that have the profiler tag.

:::{admonition} Tip
:class: info

Use this profiler if there are *specific/individual functions* you want to measure the times for.
  :::


#### Profiling with PyTorch profiler

If you want to profile the training or evaluation code and track the memory and CUDA kernel launches, consider using [PyTorch profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html).
It will run the profiler for some selected step numbers, once with `CUDA_LAUNCH_BLOCKING=1`, once with `CUDA_LAUNCH_BLOCKING=0`.
The PyTorch profiler can be enabled with `--logging.profiler=pytorch` flag.
The outputs of the profiler are trace files stored in `{PATH_TO_MODEL_OUTPUT}/profiler_traces`, and can be loaded in Google Chrome by typing `chrome://tracing`.


#### Profiling with PySpy

If you want to profile the entire codebase, consider using [PySpy](https://github.com/benfred/py-spy).

Install PySpy

```bash
pip install py-spy
```

To perform the profiling, you can either specify that you want to generate a flame graph or generate a live-view of the profiler.

- flame graph: with wandb logging and our inhouse logging disabled
    ```bash
    program="ns-train nerfacto -- --vis=wandb --logging.no-enable-profiler blender-data"
    py-spy record -o {PATH_TO_OUTPUT_SVG} $program
    ```
- top-down stats: running same program configuration as above
    ```bash
    py-spy top $program
    ```
    
:::{admonition} Attention
:class: attention

In defining `program`, you will need to add an extra `--` before you specify your program's arguments.
  :::## developer_guides/debugging_tools

### index.rst

```rst
Debugging tools
====================

We document a few of the supported tooling systems and pipelines we support for debugging our models (e.g. profiling to debug speed).
As we grow, we hope to provide more updated and extensive tooling support.

.. toctree::
    :maxdepth: 1

    local_logger
    profiling
    benchmarking
```

## developer_guides/debugging_tools

### local_logger.md

# Local writer

The `LocalWriter` simply outputs numerical stats to the terminal.
You can specify additional parameters to customize your logging experience.
A skeleton of the local writer config is defined below.

```python
"""nerfstudio/configs/base_config.py""""

@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    enable: bool = False
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        ...
    )
    max_log_size: int = 10

```

You can customize the local writer by editing the attributes:
- `enable`: enable/disable the logger.
- `stats_to_track`: all the stats that you want to print to the terminal (see list under `EventName` in `utils/writer.py`). You can add or remove any of the defined enums.
- `max_log_size`: how much content to print onto the screen (By default, only print 10 lines onto the screen at a time). If 0, will print everything without deleting any previous lines.

:::{admonition} Tip
:class: info

If you want to create a new stat to track, simply add the stat name to the `EventName` enum.
- Remember to call some put event (e.g. `put_scalar` from `utils/writer.py` to place the value in the `EVENT_STORAGE`. 
- Remember to add the new enum to the `stats_to_track` list
  :::

The local writer is easily configurable via CLI.
A few common commands to use:

- Disable local writer
    ```bash
    ns-train {METHOD_NAME} --logging.local-writer.no-enable
    ```

- Disable line wrapping
    ```bash
    ns-train {METHOD_NAME} --logging.local-writer.max-log-size=0
    ```## reference

### contributing.md

# Contributing

**💝 We're excited to have you join the nerfstudio family 💝**

We welcome community contributions to Nerfstudio! Whether you want to fix bugs, improve the documentation, or introduce new features, we appreciate your input.

Bug fixes and documentation improvements are highly valuable to us. If you come across any bugs or find areas where the documentation can be enhanced, please don't hesitate to submit a pull request (PR) with your proposed changes. We'll gladly review and integrate them into the project.

For larger feature additions, we kindly request you to reach out to us on [Discord](https://discord.gg/uMbNqcraFc) in the `#contributing` channel and create an issue on GitHub. This will allow us to discuss the feature in more detail and ensure that it aligns with the goals and direction of the repository. We cannot guarantee that the feature will be added to Nerfstudio.

In addition to code contributions, we also encourage contributors to add their own methods to our documentation. For more information on how to contribute new methods, please refer to the documentation [here](../developer_guides/new_methods.md).

## Overview

Below are the various tooling features our team uses to maintain this codebase.

| Tooling              | Support                                                    |
| -------------------- | ---------------------------------------------------------- |
| Formatting & Linting | [Ruff](https://beta.ruff.rs/docs/)                         |
| Type checking        | [Pyright](https://github.com/microsoft/pyright)            |
| Testing              | [pytest](https://docs.pytest.org/en/7.1.x/)                |
| Docs                 | [Sphinx](https://www.sphinx-doc.org/en/master/)            |
| Docstring style      | [Google](https://google.github.io/styleguide/pyguide.html) |
| JS Linting           | [eslint](https://eslint.org/)                              |

## Requirements

To install the required packages and register the pre-commit hook:

```bash
pip install -e .[dev]
pip install -e .[docs]
pre-commit install
```

This will ensure you have the required packages to run the tests, linter, build the docs, etc.
The pre-commit hook will ensure your commits comply with the repository's code style rules.

You may also need to install [pandoc](https://pandoc.org/). If you are using `conda` you can run the following:

```bash
conda install -c conda-forge pandoc
```

## Committing code

1. Make your modifications ✏️
2. Perform local checks ✅

   To ensure that you will be passing all tests and checks on github, you will need to run the following command:

   ```bash
   ns-dev-test
   ```

   This will perform the following checks and actions:

   - Formatting and linting: Ensures code is consistently and properly formatted.
   - Type checking: Ensures static type safety.
   - Pytests: Runs pytests locally to make sure added code does not break existing logic.
   - Documentation build: Builds docs locally. Ensures changes do not result in warnings/errors.
   - Licensing: Automatically adds licensing headers to the correct files.

   :::{admonition} Attention
   :class: attention
   In order to merge changes to the code base, all of these checks must be passing. If you pass these tests locally, you will likely pass on github servers as well (results in a green checkmark next to your commit).
   :::

3. Open pull request! 💌

:::{admonition} Note
:class: info

We will not review the pull request until it is passing all checks.
:::

## Maintaining documentation

### Building

Run the following to build the documentation:

```bash
python nerfstudio/scripts/docs/build_docs.py
```

:::{admonition} Tip
:class: info

- Rerun `make html` when documentation changes are made
- `make clean` is necessary if the documentation structure changes.
  :::

### Auto build

As you change or add models/components, the auto-generated [Reference API](https://docs.nerf.studio/reference/api/index.html) may change.
If you want the code to build on save you can use [sphinx autobuild](https://github.com/executablebooks/sphinx-autobuild).

:::{admonition} Tip
:class: info

If changes to the structure are made, the build files may be incorrect.
:::

```bash
pip install sphinx-autobuild
sphinx-autobuild docs docs/_build/html
```

### Adding notebooks

We support jupyter notebooks in our documentation. To improve the readability, the following custom tags can be added to the top of each code cell to hide or collapse the code.

| Tag           | Effect                                               |
| ------------- | ---------------------------------------------------- |
| # HIDDEN      | Hide code block and output                           |
| # COLLAPSED   | Collapse the code in a dropdown but show the results |
| # OUTPUT_ONLY | Only show the cell's output                          |
## reference/cli

### ns_install_cli.md

# ns-install-cli

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.completions.install
    :func: get_parser_fn
    :prog: ns-install-cli
    :nodefault:
```
## reference/cli

### ns_download_data.md

# ns-download-data

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.downloads.download_data
    :func: get_parser_fn
    :prog: ns-download-data
    :nodefault:
```
## reference/cli

### ns_render.md

# ns-render

:::{admonition} Note
:class: warning
Make sure to have [FFmpeg](https://ffmpeg.org/download.html) installed.
:::

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.render
    :func: get_parser_fn
    :prog: ns-render
    :nodefault:
```
## reference/cli

### ns_export.md

# ns-export

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.exporter
    :func: get_parser_fn
    :prog: ns-export
    :nodefault:
```
## reference/cli

### index.md

# CLI

We provide a command line interface for training your own NeRFs (no coding necessary). You can learn more about each command by using the `--help` argument.

## Commands

Here are the popular commands that we offer. If you've cloned the repo, you can also look at the [pyproject.toml file](https://github.com/nerfstudio-project/nerfstudio/blob/main/pyproject.toml) at the `[project.scripts]` section for details.

| Command                              | Description                            | Filename                                      |
| ------------------------------------ | -------------------------------------- | --------------------------------------------- |
| [ns-install-cli](ns_install_cli)     | Install tab completion for all scripts | nerfstudio/scripts/completions/install.py     |
| [ns-process-data](ns_process_data)   | Generate a dataset from your own data  | nerfstudio/scripts/process_data.py            |
| [ns-download-data](ns_download_data) | Download existing captures             | nerfstudio/scripts/downloads/download_data.py |
| [ns-train](ns_train)                 | Generate a NeRF                        | nerfstudio/scripts/train.py                   |
| [ns-viewer](ns_viewer)               | View a trained NeRF                    | nerfstudio/scripts/viewer/run_viewer.py       |
| [ns-eval](ns_eval)                   | Run evaluation metrics for your Model  | nerfstudio/scripts/eval.py                    |
| [ns-render](ns_render)               | Render out a video of your NeRF        | nerfstudio/scripts/render.py                  |
| [ns-export](ns_export)               | Export a NeRF into other formats       | nerfstudio/scripts/exporter.py                |

```{toctree}
:maxdepth: 1
:hidden:

ns_install_cli
ns_process_data
ns_download_data
ns_train
ns_render
ns_viewer
ns_export
ns_eval
```
## reference/cli

### ns_train.md

# ns-train

Primary interface for training a NeRF model. `--help` is your friend when navigating command arguments. We also recommend installing the tab completion `ns-install-cli`.

```bash
usage: ns-train {method} [method args] {dataparser} [dataparser args]
```

If you are using a nerfstudio data set, the minimal command is:

```bash
ns-train nerfacto --data YOUR_DATA
```

To learn about the available methods:

```bash
ns-train --help
```

To learn about a methods parameters:

```bash
ns-train {method} --help
```

By default the nerfstudio dataparser is used. If you would like to use a different dataparser it can be specified after all of the method arguments. For a list of dataparser options:

```bash
ns-train {method} {dataparser} --help
```
## reference/cli

### ns_process_data.md

# ns-process-data

:::{admonition} Note
:class: warning
Make sure to have [COLMAP](https://colmap.github.io) and [FFmpeg](https://ffmpeg.org/download.html) installed.  
You may also want to install [hloc](https://github.com/cvg/Hierarchical-Localization) (optional) for more feature detector and matcher options.
:::

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.process_data
    :func: get_parser_fn
    :prog: ns-process-data
    :nodefault:
```
## reference/cli

### ns_eval.md

# ns-eval

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.eval
    :func: get_parser_fn
    :prog: ns-eval
    :nodefault:
```
## reference/cli

### ns_viewer.md

# ns-viewer

```{eval-rst}
.. argparse::
    :module: nerfstudio.scripts.viewer.run_viewer
    :func: get_parser_fn
    :prog: ns-viewer
    :nodefault:
```
## reference/api

### config.rst

```rst
.. _configs:

Configs
============

.. automodule:: nerfstudio.configs.base_config
   :members:
   :show-inheritance:

```

## reference/api

### fields.rst

```rst
.. _fields:

Fields
============

Base
----------------

.. automodule:: nerfstudio.fields.base_field
   :members:
   :show-inheritance:

Density
----------------

.. automodule:: nerfstudio.fields.density_fields
   :members:
   :show-inheritance:

Nerfacto
----------------

.. automodule:: nerfstudio.fields.nerfacto_field
   :members:
   :show-inheritance:

Nerf-W
----------------

.. automodule:: nerfstudio.fields.nerfw_field
   :members:
   :show-inheritance:

SDF
----------------

.. automodule:: nerfstudio.fields.sdf_field
   :members:
   :show-inheritance:

Semantic NeRF
----------------

.. automodule:: nerfstudio.fields.semantic_nerf_field
   :members:
   :show-inheritance:

TensoRF
----------------

.. automodule:: nerfstudio.fields.tensorf_field
   :members:
   :show-inheritance:

Vanilla NeRF
----------------

.. automodule:: nerfstudio.fields.vanilla_nerf_field
   :members:
   :show-inheritance:

```

## reference/api

### viewer.rst

```rst
.. _viewer:

Viewer
============

.. automodule:: nerfstudio.viewer.viewer_elements
   :members:
   :show-inheritance:

```

## reference/api

### models.rst

```rst
.. _graphs:

Models
============

Base
----------------

.. automodule:: nerfstudio.models.base_model
   :members:
   :show-inheritance:

Instant NGP
----------------

.. automodule:: nerfstudio.models.instant_ngp
   :members:
   :show-inheritance:

Semantic NeRF-W
----------------

.. automodule:: nerfstudio.models.semantic_nerfw
   :members:
   :show-inheritance:

NeRF
----------------

.. automodule:: nerfstudio.models.vanilla_nerf
   :members:
   :show-inheritance:
```

## reference/api

### index.rst

```rst
.. _reference:

API
============

TODO: Explanation of each component

.. toctree::

    cameras
    config
    data/index
    fields
    field_components/index
    models
    model_components/index
    optimizers
    plugins
    utils/index
    viewer

```

## reference/api

### cameras.rst

```rst
.. _cameras:

Cameras
============

Cameras
----------------

.. automodule:: nerfstudio.cameras.cameras
   :members:
   :show-inheritance:

Camera Optimizers
-----------------

.. automodule:: nerfstudio.cameras.camera_optimizers
   :members:
   :show-inheritance:

Camera Paths
----------------

.. automodule:: nerfstudio.cameras.camera_paths
   :members:
   :show-inheritance:

Camera Utils
----------------

.. automodule:: nerfstudio.cameras.camera_utils
   :members:
   :show-inheritance:

Lie Groups
----------------

.. automodule:: nerfstudio.cameras.lie_groups
   :members:
   :show-inheritance:

Rays
----------------

.. automodule:: nerfstudio.cameras.rays
   :members:
   :show-inheritance:

```

## reference/api

### plugins.rst

```rst
.. _plugins:

Plugins
============

Method Registry
----------------------

.. automodule:: nerfstudio.plugins.registry
   :members:
   :show-inheritance:

DataParser Registry
----------------------

.. automodule:: nerfstudio.plugins.registry_dataparser
   :members:
   :show-inheritance:

Types
----------------------

.. automodule:: nerfstudio.plugins.types
   :members:
   :show-inheritance:

```

## reference/api

### optimizers.rst

```rst
.. _engine:

Engine
============

Optimizers
----------------

.. automodule:: nerfstudio.engine.optimizers
   :members:
   :show-inheritance:

Schedulers
----------------

.. automodule:: nerfstudio.engine.schedulers
   :members:
   :show-inheritance:

Trainer
----------------

.. automodule:: nerfstudio.engine.trainer
   :members:
   :show-inheritance:

Callbacks
----------------

.. automodule:: nerfstudio.engine.callbacks
   :members:
   :show-inheritance:

```

## reference/api/model_components

### ray_sampler.rst

```rst
.. _ray_sampler:

Ray Sampler
===================

.. automodule:: nerfstudio.model_components.ray_samplers
   :members:
   :show-inheritance:
```

## reference/api/model_components

### losses.rst

```rst
.. _losses:

Losses
===================

.. automodule:: nerfstudio.model_components.losses
   :members:
   :show-inheritance:
```

## reference/api/model_components

### renderers.rst

```rst
.. _renderers:

Renderers
============

.. automodule:: nerfstudio.model_components.renderers
   :members:
   :show-inheritance:

```

## reference/api/model_components

### index.rst

```rst
.. _graph_modules:

Model components
===================

.. toctree::

   ray_sampler
   losses
   renderers
```

## reference/api/field_components

### embeddings.rst

```rst
.. _embeddings:

Embeddings
===================

.. automodule:: nerfstudio.field_components.embedding
   :members:
   :show-inheritance:
```

## reference/api/field_components

### mlp.rst

```rst
.. _mlp:

MLP
===================

.. automodule:: nerfstudio.field_components.mlp
   :members:
   :show-inheritance:
```

## reference/api/field_components

### spatial_distortions.rst

```rst
.. _spatial_distortions:

Spatial Distortions
=====================

.. automodule:: nerfstudio.field_components.spatial_distortions
   :members:
   :show-inheritance:
```

## reference/api/field_components

### index.rst

```rst
.. _field_modules:

Field Modules
===================

TODO: High level description of field modules and how they connect together.

.. toctree::

   encodings
   embeddings
   field_heads
   mlp
   spatial_distortions
```

## reference/api/field_components

### encodings.rst

```rst
.. _encodings:

Encodings
===================

.. automodule:: nerfstudio.field_components.encodings
   :members:
   :show-inheritance:
```

## reference/api/field_components

### field_heads.rst

```rst
.. _field_heads:

Field Heads
===================

.. automodule:: nerfstudio.field_components.field_heads
   :members:
   :show-inheritance:
```

## reference/api/data

### datamanagers.rst

```rst
.. _datamanagers:

Datamanagers
============

Base
----------------

.. automodule:: nerfstudio.data.datamanagers.base_datamanager
   :members:
```

## reference/api/data

### utils.rst

```rst
.. _datautils:

Utils
============

Base
----------------

.. automodule:: nerfstudio.data.utils.colmap_parsing_utils
   :members:

Data
----------------

.. automodule:: nerfstudio.data.utils.data_utils
   :members:

Dataloader
----------------

.. automodule:: nerfstudio.data.utils.dataloaders
   :members:

Nerfstudio Collate
-------------------

.. automodule:: nerfstudio.data.utils.nerfstudio_collate
   :members:
```

## reference/api/data

### dataparsers.rst

```rst
.. _dataparser:

Data Parsers
============

* `Base Data Parser`_
* `ARKitScenes`_
* `Blender`_
* `D-NeRF`_
* `dycheck`_
* `Instant-NGP`_
* `Minimal`_
* `NeRF-OSR`_
* `Nerfstudio`_
* `nuScenes`_
* `Phototourism`_
* `ScanNet`_
* `SDFStudio`_
* `sitcoms3D`_


Base Data Parser
----------------

.. automodule:: nerfstudio.data.dataparsers.base_dataparser
   :members:

ARKitScenes
----------------

.. automodule:: nerfstudio.data.dataparsers.arkitscenes_dataparser
   :members:
   :show-inheritance:

Blender
----------------

.. automodule:: nerfstudio.data.dataparsers.blender_dataparser
   :members:
   :show-inheritance:

D-NeRF
----------------

.. automodule:: nerfstudio.data.dataparsers.dnerf_dataparser
   :members:
   :show-inheritance:

dycheck
----------------

.. automodule:: nerfstudio.data.dataparsers.dycheck_dataparser
   :members:
   :show-inheritance:

Instant-NGP
----------------

.. automodule:: nerfstudio.data.dataparsers.instant_ngp_dataparser
   :members:
   :show-inheritance:

Minimal
----------------

.. automodule:: nerfstudio.data.dataparsers.minimal_dataparser
   :members:
   :show-inheritance:

NeRF-OSR
----------------

.. automodule:: nerfstudio.data.dataparsers.nerfosr_dataparser
   :members:
   :show-inheritance:

Nerfstudio
----------------

.. automodule:: nerfstudio.data.dataparsers.nerfstudio_dataparser
   :members:
   :show-inheritance:

nuScenes
----------------

.. automodule:: nerfstudio.data.dataparsers.nuscenes_dataparser
   :members:
   :show-inheritance:

Phototourism
----------------

.. automodule:: nerfstudio.data.dataparsers.phototourism_dataparser
   :members:
   :show-inheritance:

ScanNet
----------------

.. automodule:: nerfstudio.data.dataparsers.scannet_dataparser
   :members:
   :show-inheritance:

SDFStudio
----------------

.. automodule:: nerfstudio.data.dataparsers.sdfstudio_dataparser
   :members:
   :show-inheritance:

sitcoms3D
----------------

.. automodule:: nerfstudio.data.dataparsers.sitcoms3d_dataparser
   :members:
   :show-inheritance:

```

## reference/api/data

### datasets.rst

```rst
.. _datasets:

Datasets
============

Base
----------------

.. automodule:: nerfstudio.data.datasets.base_dataset
   :members:

SDF Dataset
----------------

.. automodule:: nerfstudio.data.datasets.sdf_dataset
   :members:
   :show-inheritance:

Semantic Dataset
----------------

.. automodule:: nerfstudio.data.datasets.semantic_dataset
   :members:
   :show-inheritance:
```

## reference/api/data

### index.rst

```rst
.. _dataset:

Data
============

.. toctree::
   :titlesonly:

   dataparsers
   datamanagers
   datasets
   utils

Pixel Samplers
----------------

.. automodule:: nerfstudio.data.pixel_samplers
   :members:
   :show-inheritance:

Scene Box
----------------

.. automodule:: nerfstudio.data.scene_box
   :members:
   :show-inheritance:
```

## reference/api/utils

### math.rst

```rst
.. _math:

Math
============

.. automodule:: nerfstudio.utils.math
   :members:
   :show-inheritance:

```

## reference/api/utils

### colors.rst

```rst
.. _colors:

Colors
------------

.. automodule:: nerfstudio.utils.colors
   :members:
   :show-inheritance:

```

## reference/api/utils

### tensor_dataclass.rst

```rst
.. _tensor_dataclass:

TensorDataclass
=================

.. automodule:: nerfstudio.utils.tensor_dataclass
   :members:
   :show-inheritance:

```

## reference/api/utils

### colormaps.rst

```rst
.. _colormaps:

Colormaps
----------------

.. automodule:: nerfstudio.utils.colormaps
   :members:
   :show-inheritance:

```

## reference/api/utils

### index.rst

```rst
.. _utils:

Utils
===================

.. toctree::

   colors
   math
   colormaps
   tensor_dataclass
```

