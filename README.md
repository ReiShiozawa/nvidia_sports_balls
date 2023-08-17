# Project Name
## Sports_Balls_Classification
This program uses machine learning with Resnet 18 to classify 15 sports balls `(american_football baseball basketball billiard_ball bowling_ball cricket_ball football golf_ball hockey_ball hockey_puck rugby_ball shuttlecock table_tennis_ball tennis_ball volleyball)` and gives a confidence score with its answer.

![add image descrition here](direct image link here)

## The Algorithm

Add an explanation of the algorithm and how it works. Make sure to include details about how the code works, what it depends on, and any other relevant info. Add images or other descriptions for your project here. 

The machine learning model uses a database containing 15 types of sports balls. This is an interesting dataset because several images are misleading, e.g. some balls have been painted to look like other balls.

It trains with Resnet18 on the nano, providing a good balance of training time and model accuracy. Resnet18 uses 18 skippable layers before reaching the FC Layer. If the model is too inaccurate, there are larger resnet models such as Resnet 50 that can increase the accuracy but also increase the training time.

## Running this project

### Downloading and Sorting Database
1. Get database from Kaggle.
https://www.kaggle.com/datasets/samuelcortinhas/sports-balls-multiclass-image-classification

Combine the train and test folder into types of sports balls
(eg. american_football, baseball, basketball)

2. Transfer entire folder to Jetson Nano

3. Install Spiltfolders: `pip install split-folders`

4. Create a python file 
`import splitfolders`

`splitfolders.ratio('Data', output="sports_balls", seed=1337, ratio=(.8, 0.1,0.1))`

Replace `'Data'` with database directory.

### Preparing Training with Docker Container

5. After returning to the `Jetson-Inference` folder,
Create script to run docker

`echo "sudo docker run --runtime nvidia -it --rm --network host \
    --volume ~/nvdli-data:/nvdli-nano/data \
    --device /dev/video0 \
    nvcr.io/nvidia/dli/dli-nano-ai:v2.0.2-r32.7.1" > docker_dli_run.sh`

6. Make it executeable.

`chmod +x docker_dli_run.sh`

7. Run the docker: `./docker_dli_run.sh`

8. cd to `python/training/classification`

### Training the Model

9. Run `python3 train.py --model-dir=models/sports_balls data/sports_balls` to train Jetson Nano with the database. 

10. To export the model outside of docker, run `python3 onnx_export.py --model-dir=models/sports_balls` After doing so, there should be a file named `resnet18.onnx`

11. Leave Docker (ctrl + d)

### Using the Model

12. Run `NET=models/sports_balls` and `DATASET=data/sports_balls` to specify the folder of the model and dataset. 

13. Connect USB Webcam to Nano the run `imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt /dev/video0  sportsballs.mp4` to start classifing sports balls in real time.

14. Quit to end video, locate video in `/classification`
[View a video explanation here](video link)
