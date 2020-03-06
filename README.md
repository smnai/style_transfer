[![MIT License][license-shield]][license-url]

<br>
<p align="center">
  <h1 align="center">PyTorch Style Transfer</h1>

  <p align="center">
    More or less replicating <a href='https://arxiv.org/abs/1508.06576'> Gatys et al. 2015 </a> but not, like, exactly
</p>
<br>

<p align="center">
<img src="img/results/result_content_small_style_cubism_small_3_random.jpg" width="256"></img>

## Usage
Install requirements with:

``` sh
pip install -r requirements.txt
```
Put the content and style images in the `img` folder and run

``` sh
./src/style.py [CONTENT IMAGE NAME] [STYLE IMAGE NAME] 
```

with additional flags:
```
[-g STEPS] [-s IMAGE_SIZE] [-l NUM_STYLE_LAYERS] 
[-r RANDOM INITIAL IMAGE] [-a ALPHA] [-b BETA] [-n NORMALIZE INPUT]
```

To get parameter descriptions and defaults, run:

``` sh
./src/style.py --help
```
Just FYI, there's plenty of better Pytorch implementations of Gatys et al. 2015 online, this one is not very well optimized or tested: I wrote it over a weekend to test my understanding of style transfer.  
<br>

## Gallery 

<p align="center">
<img src='img/style_crumb_small.jpg' height='256'>
<img src='img/results/result_content_small_style_crumb_small.jpg' height='256'>
<br>
<img src='img/style_retro_small.jpg' height='256'>
<img src='img/results/result_content_small_style_retro_small.jpg' height='256'>
<br>
<img src='img/style_sacco_small.jpg' height='256'>
<img src='img/results/result_content_small_style_sacco_small.jpg' height='256'>
<br>
<img src='img/style_starry_small.jpg' height='256'>
<img src='img/results/result_content_small_style_starry_small.jpg' height='256'>
<br>
<img src='img/style_hokusai_small.jpg' height='256'>
<img src='img/results/result_content_small_style_hokusai_small.jpg' height='256'>
<br>
<img src='img/style_cubism_small_0.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_0_random.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_0.jpg' height='256'>
<br>
<img src='img/style_cubism_small_2.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_2_random.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_2.jpg' height='256'>
<br>
<img src='img/style_simpsons_small.jpg' height='256'>
<img src='img/results/result_content_small_style_simpsons_small_random.jpg' height='256'>
<img src='img/results/result_content_small_style_simpsons_small.jpg' height='256'>
<br>
<img src='img/style_cubism_small_3.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_3_random.jpg' height='256'>
<img src='img/results/result_content_small_style_cubism_small_3.jpg' height='256'>
<br>

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt