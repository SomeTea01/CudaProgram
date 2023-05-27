
#include <iostream>
#include <String>
#include <Windows.h>
#include "opencv2/opencv.hpp"

#define BS 30
#define ORDE_DITHER "-oderedDithering"
#define CONV_GBA "-gbaArrange"
#define NORM_VALUE "-valueNormalization"
#define NIGHT_VALUE "-nightVision"
#define THRE_SIZE 4

using std::string;
using std::cout;
using namespace cv;



const float THREMAP_ORDEREDDITERING[THRE_SIZE*THRE_SIZE] =
    {
        1 *255/17.0f,9 *255/17.0f,3 *255/17.0f,11*255/17.0f,
        13*255/17.0f,5 *255/17.0f,15*255/17.0f,7 *255/17.0f,
        4 *255/17.0f,12*255/17.0f,2 *255/17.0f,10*255/17.0f,
        16*255/17.0f,8 *255/17.0f,14*255/17.0f,6 *255/17.0f
    };
__global__ void mm_get_distribution00(uchar* datas,size_t* elem,size_t* div_step,int* size,uchar* result_array){
    unsigned long thread_x,thread_y,pos_x,pos_y;
    thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    pos_y = thread_y*(*div_step);//スレッドから画像上の位置を逆算
    pos_x = thread_x*(*elem);

    if(size[1]<=thread_x || size[0] <= thread_y) return;
    int value = datas[pos_y+pos_x+2];
    __syncthreads();
    result_array[value]++;
    __syncthreads();
    //printf("%d\n",result_array[value]);
}

__global__ void mm_oderDithering(uchar* datas,float* thre,size_t* elem,size_t* div_step,int* size)
{
    
    //printf("[%d,%d,%d]\n",(int)datas[0],(int)datas[1],(int)datas[2]);
    unsigned long thread_x,thread_y,pos_x,pos_y;
    thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    pos_y = thread_y*(*div_step);//スレッドから画像上の位置を逆算
    pos_x = thread_x*(*elem);
    //printf("%d , %d\n",(size_t)(*div_step),(size_t)(*elem));
    //printf("address:%d\n",datas);
    //printf("grid dimention:(%d,%d,%d)\n",gridDim.x,gridDim.y,gridDim.z);
    //printf("(%d,%d,%d,%d < %d,%d) => %d,%d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,HW[1],HW[0],pos_x,pos_y);
    //printf("[%d,%d,%d]\n",(int)datas[0],(int)datas[1],(int)datas[2]);
    //printf("[%d,%d,%d,%d]\n",(int)size[1],thread_x,size[0],thread_y);

    if(size[1]<=thread_x || size[0] <= thread_y) return;
    //printf("Convert at (%d,%d)\n",thread_x,thread_y);
    //printf("Convert addr (%d,%d)\n",pos_x,pos_y);
    __shared__ uchar proc_data;
    proc_data = datas[pos_y+pos_x+2];
    proc_data = (proc_data <= thre[4*(thread_y%4)+thread_x%4]) ? 0: 255;
    datas[pos_y+pos_x+2] = proc_data;

    //printf("Finish GPU1 Process");
}
__global__ void mm_convertGBA(uchar* datas,size_t* elem,size_t* div_step,int* size)
{
    int color_count = 3;
    unsigned long thread_x,thread_y,pos_x,pos_y;
    thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    pos_y = thread_y*(*div_step);//スレッドから画像上の位置を逆算
    pos_x = thread_x*(*elem);

    if(size[1]<=thread_x || size[0] <= thread_y) return;
    __shared__ uchar proc_data[3];
    uchar border = 255/color_count;
    byte level = (datas[pos_y+pos_x+2]+border-1)/border;
    uchar ans[3] = {
        255*108/360,
        level!=color_count ? level*border+25 : 255,
        //datas[poss_y+pos_x+1],
        //127,
        level*border
    };
    for(int i=0;i<3;i++){
        proc_data[i] = datas[pos_y+pos_x+i];
        proc_data[i] = ans[i];
        datas[pos_y+pos_x+i] = proc_data[i];
    }

    //printf("Finish GPU1 Process");
}
__global__ void mm_normalizeValue(uchar* datas,size_t* elem,size_t* div_step,int* size,uchar* min,uchar*max){
    //printf("%d,%d",(int)*min,(int)*max);
    unsigned long thread_x,thread_y,pos_x,pos_y;
    thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    pos_y = thread_y*(*div_step);//スレッドから画像上の位置を逆算
    pos_x = thread_x*(*elem);

    if(size[1]<=thread_x || size[0] <= thread_y) return;
    
    __shared__ uchar proc_data[3];
    __shared__ uchar _min;
    __shared__ uchar _max;

    proc_data[0] = 0;
    proc_data[2] = datas[pos_y+pos_x+2];
    _min = *min;
    _max = *max;
    float normalize_val = (((float)(proc_data[2]-_min))/(_max-_min));
    normalize_val = normalize_val<0 ? 0: normalize_val;
    normalize_val = normalize_val>1 ? 1: normalize_val;
    //printf("%d,%d,\n",_min[i],_max[i]);
    
    proc_data[2] = (int)(normalize_val*255);
    //datas[pos_y+pos_x+0] = proc_data[0];
    datas[pos_y+pos_x+2] = proc_data[2];


}
__global__ void mm_nightVision(uchar* datas,size_t* elem,size_t* div_step,int* size){
    //printf("%d,%d",(int)*min,(int)*max);
    unsigned long thread_x,thread_y,pos_x,pos_y;
    thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    thread_y = blockIdx.y * blockDim.y + threadIdx.y;
    pos_y = thread_y*(*div_step);//スレッドから画像上の位置を逆算
    pos_x = thread_x*(*elem);

    if(size[1]<=thread_x || size[0] <= thread_y) return;
    int half = (*div_step)/(*elem)/2;
    bool skip_flg = thread_x > half;
    if(skip_flg) return;
    
    uchar value = datas[pos_y+pos_x+2];
    uchar value_correction = 60;
    uchar hue_correction = 30;

    datas[pos_y+pos_x+0] =  255*68/360;
    datas[pos_y+pos_x+1] =  180;
    datas[pos_y+pos_x+2] =  value+value_correction>255 ? 255:value+value_correction;
}

Mat oderedDithering(Mat hsv_img,uchar* Dimg_datas,int* DimgHW,size_t* DimgElem,size_t* DimgStep,float*Dfloat){
    int height = hsv_img.rows;
    int width = hsv_img.cols;
    int elem = hsv_img.elemSize();

    std::cout << "Allocation memory is completed" << std::endl;
    
    Mat result = hsv_img.clone();
    int h_block = (height+BS-1)/BS;//切り上げ処理
    int w_block = (width+BS-1)/BS;//きりあげしょり
    
    std::cout << "block_count:(" << h_block << "," << w_block << ")" << std::endl;

    mm_oderDithering <<<dim3(w_block,h_block),dim3(BS,BS,1)>>> (Dimg_datas,Dfloat,DimgElem,DimgStep,DimgHW);
    cudaThreadSynchronize();
    cudaMemcpy(hsv_img.data,Dimg_datas,sizeof(uchar)*height*width*elem,cudaMemcpyDeviceToHost);
    result = hsv_img.clone();

    return result;
}
Mat convertGBA(Mat hsv_img,uchar* Dimg_datas,int* DimgHW,size_t* DimgElem,size_t* DimgStep){
    int height = hsv_img.rows;
    int width = hsv_img.cols;
    int elem = hsv_img.elemSize();
    
    Mat result = hsv_img.clone();
    int h_block = (height+BS-1)/BS;//切り上げ処理
    int w_block = (width+BS-1)/BS;//きりあげしょり

    mm_convertGBA <<<dim3(w_block,h_block),dim3(BS,BS,1)>>> (Dimg_datas,DimgElem,DimgStep,DimgHW);
    cudaThreadSynchronize();
    cudaMemcpy(hsv_img.data,Dimg_datas,sizeof(uchar)*height*width*elem,cudaMemcpyDeviceToHost);
    result = hsv_img.clone();

    return result;
}
Mat normalizeValue(Mat hsv_img,uchar* Dimg_datas,int* DimgHW,size_t* DimgElem,size_t* DimgStep,uchar* min,uchar* max,uchar* save_memory){
    int height = hsv_img.rows;
    int width = hsv_img.cols;
    int elem = hsv_img.elemSize();
    
    Mat result = hsv_img.clone();
    int h_block = (height+BS-1)/BS;//切り上げ処理
    int w_block = (width+BS-1)/BS;//切り上げ処理
    uchar distribution_value[256];

    mm_get_distribution00 <<<dim3(w_block,h_block),dim3(BS,BS,1)>>> (Dimg_datas,DimgElem,DimgStep,DimgHW,save_memory);
    cudaThreadSynchronize();
    cudaMemcpy(distribution_value,save_memory,sizeof(uchar)*256,cudaMemcpyDeviceToHost);

    //Caluclate min and max here;
    int remove_count_min = (height*width)*0.005;
    int remove_count_max = (height*width)*0.005;

    
    uchar min_thre = 0;
    uchar max_thre =255;
    bool min_flg = false;
    bool max_flg = false;

    for(int i=0;i<256;i++){
        
        //std::cout << (int)distribution_value[i] << "," << remove_count_min << "," << remove_count_max << std::endl;

        if(distribution_value[i]==0) continue;
        if(max_flg&&min_flg) break;
        if(remove_count_min-distribution_value[i] < 0 &&!min_flg){
            //std::cout << (int)distribution_value[i] << "," << (int)min_count << std::endl;
            min_thre = i;
            min_flg=true;
        }else{
            remove_count_min -= distribution_value[i];
        }
        if(remove_count_max-distribution_value[255-i] < 0 &&!max_flg){
            max_thre = 255-i;
            max_flg=true;
        }else{
            remove_count_max -= distribution_value[i];
        }
    }
    if(min_thre > max_thre){
        uchar dum = min_thre;
        min_thre = max_thre;
        max_thre = min_thre;
    }
    std::cout << (int)min_thre << "," << (int)max_thre <<std::endl;
    cudaMemcpy(min,&min_thre,sizeof(uchar),cudaMemcpyHostToDevice);
    cudaMemcpy(max,&max_thre,sizeof(uchar),cudaMemcpyHostToDevice);
    //std::cout << (int)min_thre << "," << (int)max_thre << std::endl;
    
    mm_normalizeValue <<<dim3(w_block,h_block),dim3(BS,BS,1)>>> (Dimg_datas,DimgElem,DimgStep,DimgHW,min,max);
    cudaThreadSynchronize();
    cudaMemcpy(hsv_img.data,Dimg_datas,sizeof(uchar)*height*width*elem,cudaMemcpyDeviceToHost);
    result = hsv_img.clone();

    return result;
}
Mat nightVision(Mat hsv_img,uchar* Dimg_datas,int* DimgHW,size_t* DimgElem,size_t* DimgStep){
    
    int height = hsv_img.rows;
    int width = hsv_img.cols;
    int elem = hsv_img.elemSize();

    std::cout << "Allocation memory is completed" << std::endl;
    
    Mat result = hsv_img.clone();
    int h_block = (height+BS-1)/BS;//切り上げ処理
    int w_block = (width+BS-1)/BS;//きりあげしょり
    
    std::cout << "block_count:(" << h_block << "," << w_block << ")" << std::endl;

    mm_nightVision <<<dim3(w_block,h_block),dim3(BS,BS,1)>>> (Dimg_datas,DimgElem,DimgStep,DimgHW);
    cudaThreadSynchronize();
    cudaMemcpy(hsv_img.data,Dimg_datas,sizeof(uchar)*height*width*elem,cudaMemcpyDeviceToHost);
    result = hsv_img.clone();

    return result;
}

void allocateGPUAddress(char* mode,uchar** Image,size_t** Elem,size_t** Step,int** Size,float** Dfloat,int height,int width,size_t elem,void** value,size_t malloc_sizes[]){
    cudaMalloc((void**)Image,sizeof(uchar)*height*width*elem);
    cudaMalloc((void**)Elem,sizeof(size_t));
    cudaMalloc((void**)Step,sizeof(size_t));
    cudaMalloc((void**)Size,sizeof(int)*2);
    cudaMalloc((void**)Dfloat,sizeof(float)*THRE_SIZE*THRE_SIZE);
    
    
    if(strcmp(mode,NORM_VALUE)==0){
        std::cout << "malloc__:" << malloc_sizes[0] << "," << malloc_sizes[1] << std::endl;
        cudaError_t error = cudaMalloc((void**)&value[0],malloc_sizes[0]);
        if(error != 0 ){
            std::cout << "Allocation error occurred!!_0";
        }
        error = cudaMalloc((void**)&value[1],malloc_sizes[1]);
        if(error != 0 ){
            std::cout << "Allocation error occurred!!_1";
        }
        error = cudaMalloc((void**)&value[2],malloc_sizes[2]);
        if(error != 0){
            std::cout << "Allocation error occurred!!_2";
        }

    }
}
void placeDataToGPUAddress(char* mode,uchar* Dimage_data,Mat img,long data_size,size_t* Delem,size_t* elem,size_t* Dstep,size_t* step,int* Dsize,int* size,float* Dfloat)
{
    if(strcmp(mode,ORDE_DITHER)==0||strcmp(mode,CONV_GBA)==0||strcmp(mode,NORM_VALUE)==0||strcmp(mode,NIGHT_VALUE)==0) cvtColor(img,img,COLOR_BGR2HSV);
    cudaMemcpy(Dimage_data,img.data,sizeof(uchar)*data_size,cudaMemcpyHostToDevice);
    cudaMemcpy(Delem,elem,sizeof(size_t),cudaMemcpyHostToDevice);
    cudaMemcpy(Dstep,step,sizeof(size_t),cudaMemcpyHostToDevice);
    cudaMemcpy(Dsize,size,sizeof(int)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(Dfloat,THREMAP_ORDEREDDITERING,sizeof(float)*THRE_SIZE*THRE_SIZE,cudaMemcpyHostToDevice);
}
void freeGPUAddress(uchar* Dimage_data,size_t* Delem,size_t* Dstep,int* Dsize,float* Dfloat)
{

    cudaFree(Dimage_data);
    cudaFree(Delem);
    cudaFree(Dsize);
    cudaFree(Dstep);
    cudaFree(Dfloat);
}

Mat ConvertPicture(Mat img,char* mode,bool is_disp,char stop_key){
    //std::cout << "Image Created" << std::endl;
    cvtColor(img,img,COLOR_BGR2HSV);
    //std::cout << hsv_img.step << "," << hsv_img.elemSize() << std::endl;

        int height = img.rows;
        int width = img.cols;
        size_t step = img.step;
        size_t elem = img.elemSize();
        int size[2]={height,width};
        uchar* Dimg_datas;
        int* DimgHW;
        size_t* DimgElem;
        size_t* DimgStep;
        float* Dfloat;
        void* Dvalues[3];
        size_t mallocSizes[3];
        if(strcmp(mode,NORM_VALUE)==0){
            mallocSizes[0]=sizeof(int);
            mallocSizes[1]=sizeof(int);
        }

        allocateGPUAddress(mode,&Dimg_datas,&DimgElem,&DimgStep,&DimgHW,&Dfloat,height,width,elem,Dvalues,mallocSizes);
        placeDataToGPUAddress(mode,Dimg_datas,img,height*width*elem,DimgElem,&elem,DimgStep,&step,DimgHW,size,Dfloat);

    if(strcmp(mode,ORDE_DITHER)==0){
        img = oderedDithering(img,Dimg_datas,DimgHW,DimgElem,DimgStep,Dfloat);
    }else if(strcmp(mode,CONV_GBA) ==0){
        img = convertGBA(img,Dimg_datas,DimgHW,DimgElem,DimgStep);
    }
    cvtColor(img,img,COLOR_HSV2BGR);

    if(is_disp){
        //imshow("img", img);
        string window_title = "Exit to press "+ stop_key;
        imshow(window_title, img);
        char input = '\n';
        while(stop_key != input){
            input = waitKey(1);
        }
    }
    freeGPUAddress(Dimg_datas,DimgElem,DimgStep,DimgHW,Dfloat);
    return img;
}
Mat ConvertPicture(char *path,char* mode,bool is_disp,char stop_key){
    //std::cout << "Convert Started!!" << std::endl;
    string path_str = std::string::basic_string(path);

    //std::cout << path_str << std::endl;
    //std::cout << &path_str << std::endl;

    Mat img = imread(path_str);
    return ConvertPicture(img,mode,is_disp,stop_key);
}
void ConvertVideo(char* path,char*mode,bool is_disp,char stop_key,bool save_flg){

    cv::VideoCapture video(path);
		if (video.isOpened() == false) return ;
        std::cout << "video process started!!" << std::endl;
        int width = video.get(CAP_PROP_FRAME_WIDTH);
        int height = video.get(CAP_PROP_FRAME_HEIGHT);
		cv::Mat image;
        video >> image;
        size_t step = image.step;
        size_t elem = image.elemSize();
        int size[2]={height,width};

        uchar* Dimg_datas;
        int* DimgHW;
        size_t* DimgElem;
        size_t* DimgStep;
        float* Dfloat;
        void* Dvalues[3];
        size_t mallocSizes[3];

        if(strcmp(mode,NORM_VALUE)==0){
            mallocSizes[0]=sizeof(uchar);
            mallocSizes[1]=sizeof(uchar);
            mallocSizes[2]=sizeof(uchar)*256;
        }


        std::cout << "Allocated Started" << std::endl;
        allocateGPUAddress(mode,&Dimg_datas,&DimgElem,&DimgStep,&DimgHW,&Dfloat,height,width,elem,Dvalues,mallocSizes);
        std::cout << "allocated Process Done!" << std::endl;
        
        placeDataToGPUAddress(mode,Dimg_datas,image,height*width*elem,DimgElem,&elem,DimgStep,&step,DimgHW,size,Dfloat);
        std::cout << "place Data Done!" << std::endl;

        int i=0;
		while (!image.empty()) {
            if(strcmp(mode,ORDE_DITHER)==0){
                cvtColor(image,image,COLOR_BGR2HSV);
                if(i!=0) cudaMemcpy(Dimg_datas,image.data,sizeof(uchar)*height*width*elem,cudaMemcpyHostToDevice);
			    image = oderedDithering(image,Dimg_datas,DimgHW,DimgElem,DimgStep,Dfloat);
                cvtColor(image,image,COLOR_HSV2BGR);
            }else if(strcmp(mode,CONV_GBA)==0){
                cvtColor(image,image,COLOR_BGR2HSV);
                if(i!=0) cudaMemcpy(Dimg_datas,image.data,sizeof(uchar)*height*width*elem,cudaMemcpyHostToDevice);
			    image = convertGBA(image,Dimg_datas,DimgHW,DimgElem,DimgStep);
                cvtColor(image,image,COLOR_HSV2BGR);
            }else if(strcmp(mode,NORM_VALUE) == 0){
                cvtColor(image,image,COLOR_BGR2HSV);
                if(i!=0) cudaMemcpy(Dimg_datas,image.data,sizeof(uchar)*height*width*elem,cudaMemcpyHostToDevice);
			    image = normalizeValue(image,Dimg_datas,DimgHW,DimgElem,DimgStep,(uchar*)Dvalues[0],(uchar*)Dvalues[1],(uchar*)Dvalues[2]);
                cvtColor(image,image,COLOR_HSV2BGR);
            }else if(strcmp(mode,NIGHT_VALUE) == 0){
                cvtColor(image,image,COLOR_BGR2HSV);
                if(i!=0) cudaMemcpy(Dimg_datas,image.data,sizeof(uchar)*height*width*elem,cudaMemcpyHostToDevice);
			    image = nightVision(image,Dimg_datas,DimgHW,DimgElem,DimgStep);
                cvtColor(image,image,COLOR_HSV2BGR);
            }

			imshow("Image", image);
			int key = waitKey(1);
			if ((char)key == 'q') {
				break;
			}
			video >> image;
            i++;
		}
        freeGPUAddress(Dimg_datas,DimgElem,DimgStep,DimgHW,Dfloat);
        if(strcmp(mode,NORM_VALUE)==0){
            cudaFree(Dvalues[0]);
            cudaFree(Dvalues[1]);
        }
}


int main(int argc,char *argv[])
{
    std::cout << argc << std::endl;
    if(argc != 3) return;
    char* path = argv[1];
    char* mode = argv[2];
    std::cout << argv[1] << ","<< argv[2] << std::endl;
    char *ext = strrchr(path,'.');
    


    if(strcmp(ext,".mp4")==0){
            //std::cout << "mp4 process" << std::endl;
            ConvertVideo(path,mode,true,'q',false);
    }else if(strcmp(ext,".png")==0){
            //std::cout << "png process"  << std::endl;
            ConvertPicture(path,mode,true,'q');
    }else{
            //std::cout << "This Extension is not supported. (" << ext <<")" << std::endl;
    }

}






