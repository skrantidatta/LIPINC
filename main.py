
from utils import get_color_structure_frames
from model import LIPINC_model
import numpy as np
from os import listdir, path
import cv2, os, sys, argparse
import json
from datetime import datetime
import time


def parse_args():
  parser = argparse.ArgumentParser(description='Inference code to LIPINC models')
  parser.add_argument('--input_path', type=str, help='This path should be an external path point to an video file')
  parser.add_argument('--output_path', type=str, help='This path should be an external path point to result folder',default = "")
  parser.add_argument('--info_path', type=str, help='This path should be an external path point to method_info.json', default = 'Media/deepfake-o-meter/method_info.json')
  parser.add_argument('--d', action="store_true", help='Use this argument when input path is a folder')
  parser.add_argument('--device', type=str, default = 'cuda', help='Set GPU or CPU as first priority')
  parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', default= "checkpoints/FakeAv.hdf5")
  
  args = parser.parse_args()
  return args 



def get_result_description(real_p):
    if real_p <= 1 and real_p >= 0.99:
        return 'This sample is certainly real.'
    elif real_p < 0.99 and real_p >= 0.75:
        return 'This sample is likely real.'
    elif real_p < 0.75 and real_p >= 0.25:
        return 'This sample is maybe real.'
    elif real_p < 0.25 and real_p >= 0.01:
        return 'This sample is unlikely real.'
    elif real_p < 0.01 and real_p >= 0:
        return 'There is no chance that the sample is real.'
    else:
        return 'Error'


class Video_Short(Exception):
    pass


def get_result(input_path,advanced_folder_path,info,checkpoint_path):

  # record start time
  start = time.time()
  Error= None
  result = -1
  input_file_name = input_path.split('/')[-1]
  print("The Video being evaluated: ",input_file_name)
  try:
    advanced_video_folder_path = os.path.join(advanced_folder_path,input_file_name.split('.')[0])
    os.makedirs(advanced_video_folder_path,exist_ok= True)

    n_frames = 5 #number of local frames
    length_error,face,combined_frames,residue_frames,_,_ = get_color_structure_frames(n_frames,input_path)
    
    if length_error:
      raise Video_Short()
    else:
      cv2.imwrite(os.path.join(advanced_video_folder_path,'face.png'), face) 
      #saving lips image 
      for idx in range(len(combined_frames)):
            lip_image = combined_frames[idx]      
            # saving the face image as a PNG file 
            cv2.imwrite(os.path.join(advanced_video_folder_path,f'lip{idx}.png'), lip_image)
      

      combined_frames,residue_frames = np.reshape(combined_frames, (1,) + combined_frames.shape), np.reshape(residue_frames, (1,) + residue_frames.shape)
      print("Shape of Color Frames: {} and Structure Frames: {}".format(combined_frames.shape,residue_frames.shape))
      model = LIPINC_model()

      model.load_weights(checkpoint_path)

      result = model.predict([combined_frames,residue_frames])
      #real probability
      result = round(float(result[0][1]),3)
      print("Result: ",result)
  
      

  except Video_Short as e:
      Error = "Video_short"
  except Exception as e:
      Error = "no_face"
    

  # record end time
  end = time.time()


  json_text = {"Task":"", # Video, Image or Audio Task, get from method_info.json
              "Input File":"", # Input file name
              "Analytic Name":"LIPINC", # Method name
              "Analysis Date":"", # str(datetime.now())
              "Original Result":"", # result generated directly from model
              "Original Result Description":"", # Explanation of result, get from method_info.json
              "Result":{}, # {"Real Probability": n, "Fake Probability": 1-n}
              "Result Description":"", # get_result_description(result)
              "Advanced Results":"N/A", # Avaliable or Not Avaliable
              "Analytic Description":"", # Method description, get from method_info.json
              "Analysis Scope":"", # Description of nput File scope, ex: DSP-FWA is for face, NoDown is for GAN image, get from method_info.json
              "Reference":"", # Paper reference, get from method_info.json
              "Code Link":"", # Github, get from method_info.json
              "Error":"N/A", #(Optional) Catched error
              "Analysis Time in Second":"", #(Optional) Model run time
              "Device":"" #(Optional) cuda or cpu
              }



  json_text['Task'] = info['task']
  json_text['Input File'] = input_file_name
  json_text['Analytic Name'] = info['analytic_name']
  json_text['Analysis Date'] = str(datetime.now())
  json_text['Original Result Description'] = info['result_description']
  if result>=0 and result<=1:
    json_text['Original Result'] = result
    json_text['Result'] = {"Real Probability": result, "Fake Probability": round(1-result,4)}
    json_text['Result Description'] = get_result_description(result)
    json_text['Advanced Results'] = f"The images of the face and extracted lips are stored in {advanced_video_folder_path}"
  else:
    json_text['Original Result'] = 'N/A'
    json_text['Result'] = {"Real Probability": 'N/A', "Fake Probability": 'N/A'}
    json_text['Result Description'] = 'Error'
    
 
  json_text['Analytic Description'] = info['analytic_description']
  json_text['Analysis Scope'] = info['analysis_scope']
  json_text['Reference'] = info['paper_reference']
  json_text['Code Link'] = info['code_reference']
  if Error:
    json_text['Error'] = info['errors'][Error]
  json_text['Analysis Time in Second'] = round((end-start),2)
  json_text['Device'] = device


  result_path =os.path.join(output_path,"result.json") 

  # Writing to result.json
  if input_is_folder:
    with open(result_path, 'a') as json_w:
      json.dump(json_text, json_w, indent=5)
  else:
    with open(result_path, 'w') as json_w:
      json.dump(json_text, json_w, indent=5)

#####################################################################################################################
def main():
  global args,device,output_path,input_is_folder

  args = parse_args()
  
  input_path = args.input_path
  output_path = args.output_path
  input_is_folder = args.d
  checkpoint_path = args.checkpoint_path
  device = args.device
  info_path = args.info_path


  # load method_info.json
  with open(info_path, 'r') as f:
    info = json.load(f)['lipinc']

  advanced_folder_path = os.path.join(output_path,"Advanced_results") 
  os.makedirs(advanced_folder_path,exist_ok= True)
  
  if "result.json" in os.listdir(output_path):
    os.remove(os.path.join(output_path,"result.json") )

  if input_is_folder:
    if not os.path.isdir(input_path):
      raise ValueError('--must be a valid path to a directory')
    input_paths = os.listdir(input_path)

    for input_file in input_paths:
      if input_file.split('.')[-1] in ["mp4"]:
        i_path = os.path.join(input_path,input_file)
        get_result(i_path,advanced_folder_path,info,checkpoint_path)

  else:
    if not os.path.isfile(input_path):
      raise ValueError('--must be a valid path to video/image file')
    get_result(input_path,advanced_folder_path,info,checkpoint_path)
    
    

if __name__ == '__main__':
    main()





