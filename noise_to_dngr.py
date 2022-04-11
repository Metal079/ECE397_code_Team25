# -*- coding: utf-8 -*-
# import the packages
import os
import torch
from torch import nn
from torchsummary import summary
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pushbullet import Pushbullet
from time import sleep
import torchaudio
import datetime

import pandas as pd
import glob

pb = Pushbullet("o.2KuiIHBmec57iSaf8Vqha8oTxpVkc301")

TEST_AUDIO_DIR = '/home/collab/Desktop/Audio DB/'
#TEST_AUDIO_DIR = '/content/Audio DB/'
annotations_file = "/home/collab/Desktop/DNGR_machine_learning/testLabels.csv"
#annotations_file = "/content/testLabels.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050
BATCH_SIZE = 1
LEARNING_RATE = 0.0001

# Each entry corresponds to a classification, (ex. 0 = gun shot, 1 = car horn, ect..) But on MNIST 0 = 0, 1 = 1, ect..
class_mapping = {
                 0 : "gun_shot",
                 1 : "NOT_gun_shot"
}

# create labels on sound files
def create_labels(sound_files_directory):
    def load_sound_files(folder, label, mode):
        test_dir = {-1: sound_files_directory}

        # Insert mode to choose correct dir
        dir_selector = {'test': test_dir}

        #Load sound files as dictionary where each key is a column name
        for root, dirs, files in os.walk(folder):
            for file in files:
                sound_files[file] = [label, dir_selector[mode][label]]

    sound_files = {}
    load_sound_files(sound_files_directory, -1, 'test')
    dataframe = pd.DataFrame(sound_files)
    dataframe.to_csv("/home/collab/Desktop/DNGR_machine_learning/testLabels.csv")
    csv2 = pd.read_csv("/home/collab/Desktop/DNGR_machine_learning/testLabels.csv", header=None).T.to_csv("/home/collab/Desktop/DNGR_machine_learning/testLabels.csv", header=False, index=False)

# Dataset loading
class SoundDataset(Dataset):
  def __init__(self,
               annotations_file,
                audio_dir,
                transformation,
                target_sample_rate,
                num_samples,
                device):
      self.annotations = pd.read_csv(annotations_file)
      self.audio_dir = audio_dir
      self.device = device
      self.transformation = transformation.to(self.device)
      self.target_sample_rate = target_sample_rate
      self.num_samples = num_samples

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
      audio_sample_path = self._get_audio_sample_path(index)
      label = self._get_audio_sample_label(index)
      signal, sr = torchaudio.load(audio_sample_path, normalize=True)
      signal = signal.to(self.device)
      signal = self._resample_if_necessary(signal, sr)
      signal = self._mix_down_if_necessary(signal)
      signal = self._cut_if_necessary(signal)
      signal = self._right_pad_if_necessary(signal)
      signal = self.transformation(signal)
      return signal, label

  def _cut_if_necessary(self, signal):
    if signal.shape[1] > self.num_samples:
        signal = signal[:, :self.num_samples]
    return signal

  def _resample_if_necessary(self, signal, sr):
      if sr != self.target_sample_rate:
          resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(device)
          signal = resampler(signal)
      return signal

  def _mix_down_if_necessary(self, signal):
      if signal.shape[0] > 1:
          signal = torch.mean(signal, dim=0, keepdim=True)
      return signal

  def _right_pad_if_necessary(self, signal):
    length_signal = signal.shape[1]
    if length_signal < self.num_samples:
        num_missing_samples = self.num_samples - length_signal
        last_dim_padding = (0, num_missing_samples)
        signal = torch.nn.functional.pad(signal, last_dim_padding)
    return signal

  def _get_audio_sample_path(self, index):
    path = self.audio_dir + self.annotations.iloc[index, 0]
    return path

  def _get_audio_sample_label(self, index):
    return self.annotations.iloc[index, 1]

# The neural network architecture
class CNNNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    # 4 conv blocks / flatten / linear / softmax
    self.conv1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv6 = nn.Sequential(
        nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.flatten = nn.Flatten()
    #self.linear = nn.Linear(128 * 5 * 4, 2) # 10 is num of classes
    self.linear = nn.Linear(2048, 2) # 10 is num of classes
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_data):
    x = self.conv1(input_data)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.conv6(x)
    x = self.flatten(x)
    logits = self.linear(x)
    predictions = self.softmax(logits)
    return predictions

# Test trained neural network with test data
def test(model, data_loader, device, delete):
  # Used to get stats on accuracy
  model.eval()

  for input in data_loader:
    #input = input.to(device) # Send to gpu
    with torch.no_grad():
      predictions = model(input[0])
      predicted_index = predictions[0].argmax(0)
      if predicted_index.item() == 0:
          log = open("/home/collab/Desktop/Logs/ML_LOG.log", 'a')
          log.write("\n")
          log.write("AI predicts " + str(predictions[0][0] * 100) + "% gunshot, " + str(predictions[0][1] * 100) + "% not a gunshot\n")
          log.write("Detected " + str(class_mapping[predicted_index.item()]) + "\n")
          print("variable is... " + class_mapping[predicted_index.item()])
          log.write(str(datetime.datetime.now()) + ",\n")
          log.write("\n")
          
          print("Notification sent")
          
          #notifications
          
          dev = pb.get_device('Samsung SM-A515U1')
          push = dev.push_note("Caution!", "Gunshot detected within your home area")
          sleep(1)
      print("\n")
      print("AI predicts " + str(predictions[0][0] * 100) + "% gunshot, " + str(predictions[0][1] * 100) + "% not a gunshot\n")
      print("Detected " + str(class_mapping[predicted_index.item()]) + "\n")
      print(str(datetime.datetime.now()) + "\n")
      print("\n")

  if delete:
    filelist = glob.glob(os.path.join(TEST_AUDIO_DIR, "*"))
    for f in filelist:
        os.remove(f)
  print("Testing Complete! @" + str(datetime.datetime.now()))

# Initialize Network
if __name__ == "__main__":
  # Check if gpu is available
  if torch.cuda.is_available():
      device = "cuda"
  else:
      device = "cpu"
  print(f"Using device {device}")

  # Take copy of current files in Audio DB to use to detect when new files are added
  current_files = glob.glob(os.path.join(TEST_AUDIO_DIR, "*"))
  if len(current_files) == 0:
    print("error need some something to be in audio DB folder before starting")


  # Used for pre-processing sound data
  mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
  )

  # Create labels needed to load data into a dataset
  create_labels(TEST_AUDIO_DIR)

  # Sound pre-processing
  sounds_test = SoundDataset(annotations_file,
            TEST_AUDIO_DIR, 
            mel_spectrogram, 
            SAMPLE_RATE, 
            NUM_SAMPLES,
            device)
  

    
  # instantiate data loaders
  test_dataloader = DataLoader(sounds_test, batch_size=BATCH_SIZE, shuffle=True)

  # construct model and assign it to device
  cnn = CNNNetwork().to(device)

  # initialise loss funtion + optimiser
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.Adam(cnn.parameters(),
                                lr=LEARNING_RATE)

  # Load old model 
  state_dict = torch.load("/home/collab/Desktop/DNGR_machine_learning/neuralNetwork._6layers_90%.pth", map_location=torch.device('cpu'))
  cnn.load_state_dict(state_dict)
    
  # Test Model (change last parameter to True to delete files after classifiying them)
  test(cnn, test_dataloader, device, False)
  
  # Checks as often as possible for new files to be added to the folder and runs forever
  while True:
    new_files = glob.glob(os.path.join(TEST_AUDIO_DIR, "*"))
    if new_files != current_files and len(new_files) != 0:
        current_files = new_files

        #Create labels needed to load data into a dataset
        create_labels(TEST_AUDIO_DIR)

        # Sound pre-processing
        sounds_test = SoundDataset(annotations_file,
            TEST_AUDIO_DIR, 
            mel_spectrogram, 
            SAMPLE_RATE, 
            NUM_SAMPLES,
            device)
         
        # instantiate data loaders
        test_dataloader = DataLoader(sounds_test, batch_size=BATCH_SIZE, shuffle=True)
         
        # Test Model (change last parameter to True to delete files after classifiying them)
        test(cnn, test_dataloader, device, False)