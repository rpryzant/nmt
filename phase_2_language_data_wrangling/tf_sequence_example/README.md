### Introduction

This project is going to be making use of sequential data. `tf.SequenceExample` is a protobuf definition that's specifically designed for sequential inputs. I'm going to be using this in the main project, so it's good to learn it by playing with it in a minimal sandbox.

### Benefits of tf.SequenceExample

* breaking data into this format lets you leverage tf's built-in distributed training support
* the model is rendered independant of its data (anything can be stored in `tf.SequenceExample` if you want to reuse your model for something else)
* tf comes with some handy data loading functions (i.e. padding, batching) that are easy to use once you've packed your data in this way
* using `tf.SequenceExample` forces you to seperate your data processing and model code, which is always good practice


### This project

This project is pretty simple. I'm taking the model and data from phase 1 -- LSTM. Trying to do the exact same thing except with `tf.SequenceExample`.