# set -e
DATA_DIR=data/replica_v1

for p in {a..q}
do
  # Ensure files are continued in case the script gets interrupted halfway through
  wget --continue https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0/replica_v1_0.tar.gz.parta$p
done

# Create the destination directory if it doesn't exist yet
mkdir -p ${DATA_DIR}

cat replica_v1_0.tar.gz.part?? | unpigz -p 32  | tar -xvC ${DATA_DIR}

#download, unzip, and merge the additional habitat configs
wget http://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip -P assets/
unzip -qn assets/additional_habitat_configs.zip -d ${DATA_DIR}



# Define the base directory containing the SCENE directories.

# Iterate over each SCENE directory within the base directory.
for scene_dir in "$DATA_DIR"/*/; do
  # Define the source JSON file path.
  src_json="$scene_dir/habitat/replica_stage.stage_config.json"
  
  # Define the destination JSON file path.
  dest_json="$scene_dir/habitat/replicaSDK_stage.stage_config.json"
  
  # Check if the source JSON file exists.
  if [[ -f "$src_json" ]]; then
    # Copy the source JSON file to the destination.
    cp "$src_json" "$dest_json"
    
    # original file use infor_semantic.txt
    sed -i 's/"info_semantic.txt"/"info_semantic.json"/' "$dest_json"
    
    echo "Processed $dest_json"
  else
    echo "Source file does not exist: $src_json"
  fi
done

# remove downloaded zip files
for p in {a..q}
do 
    rm replica_v1_0.tar.gz.parta$p
done
