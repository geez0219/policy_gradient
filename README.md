# policy_gradient
To create deep neural network to play atari games by policy gradient
* Pong (the green part is trained AI player)

![ezgif com-video-to-gif](https://user-images.githubusercontent.com/27904418/54260725-01568800-4527-11e9-8f28-80b6547b6dfb.gif)



# How to use it 
notice:
in the following tutorial, you need to
1. run in to certain directory with name "solve_xxx" (xxx is the game name)
2. add the repository into python system path

## train model 
### runing code:
```
solve_xxx# python3 train_model.py [-h] [-l LEARNING_RATE] [-n GAMES_NUM] [-r RECORD_PERIOD] [-p SAVE_PATH] run_name
```
This command will train the game model and output the model checkpoint and logs in the result folder which looks like
* h: help
* l: learning rate
* n: the number of game play
* r: record period of tensorboard
* p: the saving path of the run_model (it will create <save_path>/<run_name> directory)

```
|- result/
  |- <run_name>/ # the model name
    |- tensorboard/ # the log files of training
    |- a bunch of check point files...
```
If there is a another result files with the same run_name while running the code, we can select either 
1. exit
2. resume the training from last check point
3. start a new training and overwrite the old one
4. rename the run name and start a new training


## test the model
### running code:
```
solve_xxx# python3 test_model.py [-h] [-n GAMES_NUM] [-p LOAD_PATH] [-s] [-c] run_name
```
This command will test the <load_path>/<run_name> model by playing several games
* h: help
* n: the number of testing game play 
* s: show the game play screen
* c: every times choose the action with highest probability rather than sample it 
* p: the saving path of the run_model (it will load <save_path>/<run_name> directory)


