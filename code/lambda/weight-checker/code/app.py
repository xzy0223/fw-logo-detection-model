import json
import os, tarfile
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

def lambda_handler(event, context):

    base_model_path = event["inputs"]["BaseTrainingResults"]["ModelArtifacts"]["S3ModelArtifacts"]
    base_model_bucket, base_model_key = base_model_path.split('//')[1].split('/', 1)
    model_local_path = '/tmp/' + base_model_key.split('/')[-1]
    s3_client.download_file(base_model_bucket, base_model_key, model_local_path)

    best_weights_bucket, best_weights_prefix = event["inputs"]["BestWeightsPath"].split('//')[1].split('/', 1)

    incre_trn_data_path = event["inputs"]["IncrementalTrainingDataPath"]
    incre_trn_weights_bucket = incre_trn_data_path.split('//')[1].split('/', 1)[0]
    incre_trn_weights_prefix = incre_trn_data_path.split('//')[1].split('/', 1)[1] + 'weights/'
    
    if event["inputs"]["IsFullTraining"]=="True":
       
        # try:
        #     s3_client.head_object(Bucket=best_pt_bucket, Key=best_pt_prefix+'best.pt')
        # except botocore.exceptions.ClientError as error:
        #     if error.response['Error']['Code'] == '404'
        #         print("No current best.pt in s3://{}/{}".format(best_pt_bucket, best_pt_prefix))
        #     else:
        #         raise error

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

        # cp new best.pt of base training to the best weight path as the neweast and current best.pt   
        tar = tarfile.open(model_local_path)
        best_pt_member = tar.getmember(event["inputs"]["BaseTrainingResults"]["HyperParameters"]["name"].strip("\\\"")+'/weights/best.pt')

        res = tar.extractfile(best_pt_member)
        s3_client.upload_fileobj(res, best_weights_bucket, best_weights_prefix+'best.pt')

    elif check_weights_file(incre_trn_weights_bucket, incre_trn_weights_prefix+'best.pt') == False:
        s3_client.copy_object(
                CopySource={
                    'Bucket': best_weights_bucket,
                    'Key': best_weights_prefix+'best.pt'
                },
                Bucket=incre_trn_weights_bucket,
                Key=incre_trn_weights_prefix+'best.pt'
            )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "best.pt has been uploaded to s3://{}/{}".format(incre_trn_weights_bucket,incre_trn_weights_prefix),
            }
        ),
    }
