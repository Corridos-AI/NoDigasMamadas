import os


def videoToFrames(path):
    folderPath = "videos"
    namePath = os.path.splitext(os.path.basename(path))[0]
    print(namePath)
    destination_folder = os.path.join(folderPath, namePath)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    source_path = path
    destination_path = os.path.join(destination_folder, os.path.basename(path))

   
    command = "cp {} {}".format(source_path, destination_path)
    os.system(command)

 
    output_directory = os.path.join(destination_folder, 'frames')
    os.makedirs(output_directory, exist_ok=True)
    os.system("ffmpeg -i {} -vf fps=1 {}/video%d.jpg".format(destination_path, output_directory))
    
    return output_directory
    
    
path = r"C:\Users\juanm\Videos\prueba\VideoCodefest_006-2min.mpg"
videoToFrames(path)
