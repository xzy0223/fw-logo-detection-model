import json
import os, tarfile
import shutil
import time
import botocore
import boto3

s3_client = boto3.client("s3")

def check_weights_file(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as error:
        if error.response['Error']['Code'] == '404':
            print("No current best.pt in s3://{}/{}".format(bucket, key.rsplit('/',1)[0]))
            return False
        else:
            raise error
    else:
        return True
    
def create_tar_gz(source_dir, output_path):
    with tarfile.open(output_path, "w:gz") as tar:
        # 遍历源目录及其一级子目录中的文件
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # 构建文件的绝对路径
                file_path = os.path.join(root, file)
                # 将文件添加到tar包中，使用相对路径作为文件名
                tar.add(file_path, arcname=os.path.relpath(file_path, source_dir))

def lambda_handler(event, context):
    """
    placeholder
    """
    output_model_path = event["inputs"]['IncrementalTrainingResults']['ModelArtifacts']['S3ModelArtifacts']
    output_model_bucket, output_model_key = output_model_path.split('//')[1].split('/', 1)
    model_local_path = '/tmp/' + output_model_key.split('/')[-1]
    s3_client.download_file(output_model_bucket, output_model_key, model_local_path)

    # cp new best.pt of base training to the best weight path as the neweast and current best.pt  
    best_weights_bucket, best_weights_prefix = event["inputs"]["BestWeightsPath"].split('//')[1].split('/', 1) 
    tar = tarfile.open(model_local_path)
    best_pt_member = tar.getmember(event["inputs"]["BaseTrainingResults"]["HyperParameters"]["name"].strip("\\\"")+'/weights/best.pt')
    buffered_writer = tar.extractfile(best_pt_member)

    # if have best.pt, create a backup of it
    if check_weights_file(best_weights_bucket, best_weights_prefix+'best.pt'):
            s3_client.copy_object(
                CopySource={
                    'Bucket': best_weights_bucket,
                    'Key': best_weights_prefix+'best.pt'
                },
                Bucket=best_weights_bucket,
                Key=best_weights_prefix+'best_'+str(int(time.time()))+'.pt'
            )
    # upload updated best.pt to best weight path
    s3_client.upload_fileobj(buffered_writer, best_weights_bucket, best_weights_prefix+'best.pt')

    inference_model_local_path = '/tmp/model/'
    os.mkdir(inference_model_local_path)
    os.mkdir(inference_model_local_path+'code/')
    inference_model_file_local_path = inference_model_local_path +'best.pt'

    tar = tarfile.open(model_local_path)
    best_pt_member = tar.getmember(event["inputs"]["BaseTrainingResults"]["HyperParameters"]["name"].strip("\\\"")+'/weights/best.pt')
    buffered_writer = tar.extractfile(best_pt_member)   

    with open(inference_model_file_local_path, 'wb') as file:
        shutil.copyfileobj(buffered_writer, file)
    buffered_writer.close()
    print('/tmp/model/best.pt done')

    inference_script_path = event["inputs"]['InferenceCodePath']
    inference_script_bucket, inference_script_prefix = inference_script_path.split("//")[1].split('/', 1)

    model_path = event["inputs"]['ModelPath']
    model_bucket, model_prefix = model_path.split("//")[1].split('/', 1)

    response = s3_client.list_objects_v2(
        Bucket=inference_script_bucket,
        Prefix=inference_script_prefix
    )

    for obj in response["Contents"]:
        file_name = obj["Key"].rsplit('/',1)[-1]
        if file_name in ["inference.py", "requirements.txt"]:
            print()
            s3_client.download_file(inference_script_bucket, obj["Key"], inference_model_local_path+'code/'+file_name)

    create_tar_gz(inference_model_local_path, '/tmp/model_final.tar.gz')

    s3_client.upload_file('/tmp/model_final.tar.gz', model_bucket, model_prefix+'model.tar.gz')

    return {
        "statusCode": 200,
        "ModelPath": "s3://{}/{}model.tar.gz".format(model_bucket, model_prefix)
    }
