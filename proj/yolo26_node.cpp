
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>
#include <cv_bridge/cv_bridge.h>
#include <cstring>
#include <cstdio>
#include <cctype>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <unistd.h>

// ================= TensorRT Logger =================

class TRTLogger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

// ================= YOLO Detection =================

struct Detection
{
    cv::Rect2d box;
    float confidence;
    int class_id;
};

// ================= YOLO ROS Node =================

class Yolo26Node
{
public:

    Yolo26Node()
        : nh_(), pnh_("~")
    {
        // ROS 参数
        pnh_.param<std::string>(
            "image_topic", image_topic_, "/oak/rgb/image_raw");

        pnh_.param<std::string>(
            "detection_topic", detection_topic_, "/yolo/detections_cpp");

        pnh_.param<std::string>(
            "annotated_image_topic", annotated_image_topic_, "/yolo/image_cpp");

        pnh_.param<std::string>(
            "model_path", model_path_,
            "/home/nvidia/auto/lidar_camera_ws/yolo26n.engine");

        pnh_.param<int>("image_size", image_size_, 640);
        pnh_.param<double>("confidence", confidence_, 0.25);
        pnh_.param<double>("iou", iou_, 0.45);
        pnh_.param<double>("max_rate", max_rate_, 0.0);

        if (image_size_ != 640)
            throw std::runtime_error(
                "当前 Engine 输入固定为 640x640，image_size 必须为 640");

        if (!std::isfinite(confidence_) || !std::isfinite(iou_) ||
            !std::isfinite(max_rate_) ||
            confidence_ < 0.0 || confidence_ > 1.0 ||
            iou_ < 0.0 || iou_ > 1.0 || max_rate_ < 0.0)
            throw std::runtime_error("confidence、iou 或 max_rate 参数无效");

        // 官方 YOLO26n COCO 80 类名称
        class_names_ = {
            "person", "bicycle", "car", "motorcycle", "airplane",
            "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut",
            "cake", "chair", "couch", "potted plant", "bed",
            "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        // 加载 TensorRT Engine
        loadEngine();

        // ROS 发布订阅
        detection_pub_ =
            nh_.advertise<vision_msgs::Detection2DArray>(
                detection_topic_, 1);

        image_pub_ =
            nh_.advertise<sensor_msgs::Image>(
                annotated_image_topic_, 1);

        image_sub_ = nh_.subscribe(
            image_topic_,
            1,
            &Yolo26Node::imageCallback,
            this,
            ros::TransportHints().tcpNoDelay());

        last_log_time_ = std::chrono::steady_clock::now();

        ROS_INFO("YOLO26 C++ 节点启动成功");
        ROS_INFO("Engine: %s", model_path_.c_str());
        ROS_INFO("Image topic: %s", image_topic_.c_str());
        ROS_INFO("Detection topic: %s", detection_topic_.c_str());
        ROS_INFO("Annotated image topic: %s",
                 annotated_image_topic_.c_str());
        ROS_INFO("Input: 640x640, Batch=1, FP32 I/O");
        ROS_INFO("Confidence=%.2f, IoU=%.2f, Max rate=%.2f",
                 confidence_, iou_, max_rate_);
    }

    ~Yolo26Node()
    {
        if (device_input_) cudaFree(device_input_);
        if (device_output_) cudaFree(device_output_);

        if (stream_) cudaStreamDestroy(stream_);

        delete context_;
        delete engine_;
        delete runtime_;
    }

private:

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber image_sub_;
    ros::Publisher detection_pub_;
    ros::Publisher image_pub_;

    std::string image_topic_;
    std::string detection_topic_;
    std::string annotated_image_topic_;
    std::string model_path_;

    int image_size_;
    double confidence_;
    double iou_;
    double max_rate_;

    TRTLogger logger_;

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    cudaStream_t stream_ = nullptr;

    void* device_input_ = nullptr;
    void* device_output_ = nullptr;

    int input_index_ = -1;
    int output_index_ = -1;
    std::string input_name_;
    std::string output_name_;

    int output_channels_ = 0;
    int output_candidates_ = 0;
    bool end_to_end_output_ = false;

    std::vector<float> input_buffer_;
    std::vector<float> output_buffer_;

    std::vector<std::string> class_names_;

    bool has_last_start_ = false;
    std::chrono::steady_clock::time_point last_inference_start_;
    std::chrono::steady_clock::time_point last_log_time_;

    double interval_sum_ = 0.0;
    int interval_count_ = 0;
    int frame_count_ = 0;

    // ================= Engine Loading =================

    void loadEngine()
    {
        std::ifstream file(model_path_, std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("无法打开 Engine 文件");

        file.seekg(0, std::ios::end);
        const std::streamoff file_size = file.tellg();
        if (file_size <= 0)
            throw std::runtime_error("Engine 文件为空或无法读取");

        // Ultralytics Engine 前面有 4 字节 metadata 长度和 JSON；
        // trtexec 等工具生成的纯 TensorRT plan 则从文件开头读取。
        uint32_t metadata_size = 0;
        std::streamoff engine_offset = 0;
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char*>(&metadata_size), 4);
        if (file.gcount() == static_cast<std::streamsize>(sizeof(metadata_size)) &&
            metadata_size > 0 &&
            static_cast<std::streamoff>(metadata_size) < file_size - 4)
        {
            std::vector<char> metadata(metadata_size);
            file.read(metadata.data(), static_cast<std::streamsize>(metadata.size()));
            if (file.good())
            {
                const auto first_char = std::find_if(
                    metadata.begin(), metadata.end(),
                    [](unsigned char c) { return !std::isspace(c); });
                if (first_char != metadata.end() && *first_char == '{')
                    engine_offset = 4 + static_cast<std::streamoff>(metadata_size);
            }
        }

        file.clear();
        file.seekg(engine_offset, std::ios::beg);
        std::vector<char> engine_data(
            static_cast<size_t>(file_size - engine_offset));
        file.read(engine_data.data(), static_cast<std::streamsize>(engine_data.size()));

        if (file.gcount() != static_cast<std::streamsize>(engine_data.size()))
            throw std::runtime_error("Engine 数据读取失败");

        runtime_ = nvinfer1::createInferRuntime(logger_);

        if (!runtime_)
            throw std::runtime_error("TensorRT Runtime 创建失败");

        engine_ = runtime_->deserializeCudaEngine(
            engine_data.data(), engine_data.size());

        if (!engine_)
            throw std::runtime_error("TensorRT Engine 反序列化失败");

        context_ = engine_->createExecutionContext();

        if (!context_)
            throw std::runtime_error("Execution Context 创建失败");

        if (cudaStreamCreate(&stream_) != cudaSuccess)
            throw std::runtime_error("CUDA Stream 创建失败");

        nvinfer1::Dims input_dims{};
        nvinfer1::Dims output_dims{};
        nvinfer1::DataType input_type;
        nvinfer1::DataType output_type;

#if NV_TENSORRT_MAJOR >= 10
        int input_count = 0;
        int output_count = 0;
        const int tensor_count = engine_->getNbIOTensors();
        for (int i = 0; i < tensor_count; ++i)
        {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (!tensor_name)
                continue;

            if (engine_->getTensorIOMode(tensor_name) ==
                nvinfer1::TensorIOMode::kINPUT)
            {
                input_name_ = tensor_name;
                ++input_count;
            }
            else
            {
                output_name_ = tensor_name;
                ++output_count;
            }
        }

        if (input_count != 1 || output_count != 1)
            throw std::runtime_error("Engine 必须恰好包含一个输入和一个输出");

        input_dims = engine_->getTensorShape(input_name_.c_str());
        if (input_dims.nbDims == 4 &&
            (input_dims.d[0] < 0 || input_dims.d[1] < 0 ||
             input_dims.d[2] < 0 || input_dims.d[3] < 0))
        {
            nvinfer1::Dims requested_shape{};
            requested_shape.nbDims = 4;
            requested_shape.d[0] = 1;
            requested_shape.d[1] = 3;
            requested_shape.d[2] = 640;
            requested_shape.d[3] = 640;
            if (!context_->setInputShape(input_name_.c_str(), requested_shape))
                throw std::runtime_error("无法将 Engine 输入设置为 1x3x640x640");
        }

        input_dims = context_->getTensorShape(input_name_.c_str());
        output_dims = context_->getTensorShape(output_name_.c_str());
        input_type = engine_->getTensorDataType(input_name_.c_str());
        output_type = engine_->getTensorDataType(output_name_.c_str());
#else
        const int binding_count = engine_->getNbBindings();
        int input_count = 0;
        int output_count = 0;
        for (int i = 0; i < binding_count; ++i)
        {
            const char* binding_name = engine_->getBindingName(i);
            if (engine_->bindingIsInput(i))
            {
                input_index_ = i;
                input_name_ = binding_name ? binding_name : "";
                ++input_count;
            }
            else
            {
                output_index_ = i;
                output_name_ = binding_name ? binding_name : "";
                ++output_count;
            }
        }

        if (binding_count != 2 || input_count != 1 || output_count != 1)
            throw std::runtime_error("Engine 必须恰好包含一个输入和一个输出");

        input_dims = engine_->getBindingDimensions(input_index_);
        if (input_dims.nbDims == 4 &&
            (input_dims.d[0] < 0 || input_dims.d[1] < 0 ||
             input_dims.d[2] < 0 || input_dims.d[3] < 0))
        {
            nvinfer1::Dims requested_shape{};
            requested_shape.nbDims = 4;
            requested_shape.d[0] = 1;
            requested_shape.d[1] = 3;
            requested_shape.d[2] = 640;
            requested_shape.d[3] = 640;
            if (!context_->setBindingDimensions(input_index_, requested_shape))
                throw std::runtime_error("无法将 Engine 输入设置为 1x3x640x640");
        }

        input_dims = context_->getBindingDimensions(input_index_);
        output_dims = context_->getBindingDimensions(output_index_);
        input_type = engine_->getBindingDataType(input_index_);
        output_type = engine_->getBindingDataType(output_index_);
        if (!context_->allInputDimensionsSpecified())
            throw std::runtime_error("Engine 输入维度未完全指定");
#endif

        if (input_dims.nbDims != 4 ||
            input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] != 640 || input_dims.d[3] != 640)
            throw std::runtime_error("Engine 输入 Shape 不符合预期，要求 [1,3,640,640]");

        if (output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
            output_dims.d[1] <= 0 || output_dims.d[2] <= 0)
            throw std::runtime_error("Engine 输出 Shape 无效");

        if (input_type != nvinfer1::DataType::kFLOAT ||
            output_type != nvinfer1::DataType::kFLOAT)
            throw std::runtime_error("当前代码要求 Engine 输入输出为 FP32");

        // YOLO26 常见输出有原始 one-to-many [1,84,N] 和
        // NMS-free [1,N,6] 两种，不能共用同一套解码逻辑。
        if (output_dims.d[1] == 84)
        {
            output_channels_ = output_dims.d[1];
            output_candidates_ = output_dims.d[2];
            end_to_end_output_ = false;
        }
        else if (output_dims.d[2] == 6)
        {
            output_candidates_ = output_dims.d[1];
            output_channels_ = output_dims.d[2];
            end_to_end_output_ = true;
        }
        else
        {
            throw std::runtime_error(
                "不支持的 YOLO26 输出 Shape；预期 [1,84,N] 或 [1,N,6]");
        }

        if (!end_to_end_output_ &&
            static_cast<int>(class_names_.size()) != output_channels_ - 4)
            throw std::runtime_error("类别名称数量与 Engine 不匹配");

        const size_t input_size =
            3ULL * 640 * 640 * sizeof(float);

        const size_t output_size =
            static_cast<size_t>(output_channels_) *
            output_candidates_ * sizeof(float);

        if (cudaMalloc(&device_input_, input_size) != cudaSuccess)
            throw std::runtime_error("CUDA 输入内存分配失败");

        if (cudaMalloc(&device_output_, output_size) != cudaSuccess)
            throw std::runtime_error("CUDA 输出内存分配失败");

#if NV_TENSORRT_MAJOR >= 10
        if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
            !context_->setTensorAddress(output_name_.c_str(), device_output_))
            throw std::runtime_error("TensorRT Tensor 地址设置失败");
#endif

        input_buffer_.resize(3 * 640 * 640);
        output_buffer_.resize(
            output_channels_ * output_candidates_);

        ROS_INFO("TensorRT Engine 加载成功");
        ROS_INFO("Input tensor: %s", input_name_.c_str());
        ROS_INFO("Output tensor: %s", output_name_.c_str());
        if (end_to_end_output_)
            ROS_INFO("Output shape: [1, %d, 6] (end-to-end)",
                     output_candidates_);
        else
            ROS_INFO("Output shape: [1, %d, %d] (raw)",
                     output_channels_, output_candidates_);
    }

    // ================= Image Preprocessing =================

    void preprocess(
        const cv::Mat& image,
        int& pad_left,
        int& pad_top,
        float& scale)
    {
        if (image.empty() || image.rows <= 0 || image.cols <= 0 ||
            image.type() != CV_8UC3 || image.data == nullptr)
            throw std::runtime_error(
                "输入图像无效；预期非空的 CV_8UC3/bgr8 图像");

        const size_t min_row_bytes =
            static_cast<size_t>(image.cols) * image.elemSize();
        if (image.step[0] < min_row_bytes)
            throw std::runtime_error("输入图像 step 小于一行图像数据长度");

        const int original_width = image.cols;
        const int original_height = image.rows;

        scale = std::min(
            640.0f / original_width,
            640.0f / original_height);

        const int new_width =
            static_cast<int>(std::round(original_width * scale));

        const int new_height =
            static_cast<int>(std::round(original_height * scale));

        if (!std::isfinite(scale) || scale <= 0.0f ||
            new_width <= 0 || new_height <= 0 ||
            new_width > 640 || new_height > 640)
            throw std::runtime_error("Letterbox 缩放尺寸无效");

        cv::Mat resized;
        cv::resize(
            image,
            resized,
            cv::Size(new_width, new_height),
            0,
            0,
            cv::INTER_LINEAR);

        const int dw = 640 - new_width;
        const int dh = 640 - new_height;

        pad_left = static_cast<int>(std::round(dw / 2.0 - 0.1));
        const int pad_right = dw - pad_left;

        pad_top = static_cast<int>(std::round(dh / 2.0 - 0.1));
        const int pad_bottom = dh - pad_top;

        cv::Mat padded;

        cv::copyMakeBorder(
            resized,
            padded,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv::BORDER_CONSTANT,
            cv::Scalar(114, 114, 114));

        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

        cv::Mat float_image;
        rgb.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels;
        cv::split(float_image, channels);

        const size_t plane_size = 640 * 640;

        for (int c = 0; c < 3; ++c)
        {
            std::memcpy(
                input_buffer_.data() + c * plane_size,
                channels[c].data,
                plane_size * sizeof(float));
        }
    }

    // ================= YOLO26 Postprocessing =================

    std::vector<Detection> postprocess(
        const cv::Mat& image,
        int pad_left,
        int pad_top,
        float scale)
    {
        const float* output = output_buffer_.data();
        std::vector<Detection> detections;

        const auto add_detection = [&](float x1, float y1, float x2, float y2,
                                       float score, int class_id)
        {
            if (!std::isfinite(x1) || !std::isfinite(y1) ||
                !std::isfinite(x2) || !std::isfinite(y2) ||
                !std::isfinite(score))
                return;

            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows)));
            x2 = std::max(0.0f, std::min(x2, static_cast<float>(image.cols)));
            y2 = std::max(0.0f, std::min(y2, static_cast<float>(image.rows)));

            if (x2 <= x1 || y2 <= y1)
                return;

            Detection det;
            det.box = cv::Rect2d(x1, y1, x2 - x1, y2 - y1);
            det.confidence = score;
            det.class_id = class_id;
            detections.push_back(det);
        };

        if (end_to_end_output_)
        {
            // [1,N,6] rows are [x1,y1,x2,y2,confidence,class_id];
            // coordinates are already xyxy and the graph has selected boxes.
            for (int i = 0; i < output_candidates_; ++i)
            {
                const float* row = output + i * 6;
                const float score = row[4];
                const float class_value = row[5];

                if (!std::isfinite(score) || score < confidence_ ||
                    !std::isfinite(class_value))
                    continue;

                const int class_id = static_cast<int>(std::round(class_value));
                if (class_id < 0 ||
                    class_id >= static_cast<int>(class_names_.size()) ||
                    std::abs(class_value - class_id) > 0.01f)
                    continue;

                add_detection(
                    (row[0] - pad_left) / scale,
                    (row[1] - pad_top) / scale,
                    (row[2] - pad_left) / scale,
                    (row[3] - pad_top) / scale,
                    score,
                    class_id);
            }
        }
        else
        {
            const int num_classes = output_channels_ - 4;
            const int num_boxes = output_candidates_;
            std::vector<std::vector<cv::Rect2d>> boxes(num_classes);
            std::vector<std::vector<float>> scores(num_classes);

            for (int i = 0; i < num_boxes; ++i)
            {
                const float cx = output[0 * num_boxes + i];
                const float cy = output[1 * num_boxes + i];
                const float w  = output[2 * num_boxes + i];
                const float h  = output[3 * num_boxes + i];

                int best_class = -1;
                float best_score = 0.0f;

                for (int c = 0; c < num_classes; ++c)
                {
                    const float score = output[(4 + c) * num_boxes + i];
                    if (std::isfinite(score) && score > best_score)
                    {
                        best_score = score;
                        best_class = c;
                    }
                }

                if (best_class < 0 || best_score < confidence_ ||
                    !std::isfinite(cx) || !std::isfinite(cy) ||
                    !std::isfinite(w) || !std::isfinite(h))
                    continue;

                // 原始 one-to-many 输出为 xywh，再还原到源图像。
                const float x1 = (cx - w * 0.5f - pad_left) / scale;
                const float y1 = (cy - h * 0.5f - pad_top) / scale;
                const float x2 = (cx + w * 0.5f - pad_left) / scale;
                const float y2 = (cy + h * 0.5f - pad_top) / scale;

                if (x2 <= x1 || y2 <= y1)
                    continue;

                boxes[best_class].emplace_back(
                    x1, y1, x2 - x1, y2 - y1);
                scores[best_class].push_back(best_score);
            }

            // 原始输出仍需按类别执行 NMS，避免不同类别互相抑制。
            for (int c = 0; c < num_classes; ++c)
            {
                if (boxes[c].empty())
                    continue;

                std::vector<int> indices;
                cv::dnn::NMSBoxes(
                    boxes[c], scores[c], static_cast<float>(confidence_),
                    static_cast<float>(iou_), indices);

                for (int index : indices)
                    add_detection(
                        static_cast<float>(boxes[c][index].x),
                        static_cast<float>(boxes[c][index].y),
                        static_cast<float>(boxes[c][index].x + boxes[c][index].width),
                        static_cast<float>(boxes[c][index].y + boxes[c][index].height),
                        scores[c][index], c);
            }
        }

        // 按置信度排序，最多保留 300 个检测结果
        std::sort(
            detections.begin(),
            detections.end(),
            [](const Detection& a, const Detection& b)
            {
                return a.confidence > b.confidence;
            });

        if (detections.size() > 300)
            detections.resize(300);

        return detections;
    }

    // ================= ROS Message Publishing =================

    void publishResults(
        const sensor_msgs::ImageConstPtr& image_msg,
        const cv::Mat& image,
        const std::vector<Detection>& detections)
    {
        vision_msgs::Detection2DArray detection_array;
        detection_array.header = image_msg->header;

        cv::Mat annotated = image.clone();

        for (const auto& det : detections)
        {
            const cv::Rect rect(
                static_cast<int>(det.box.x),
                static_cast<int>(det.box.y),
                static_cast<int>(det.box.width),
                static_cast<int>(det.box.height));

            const std::string& class_name =
                class_names_[det.class_id];

            char score_text[32];
            std::snprintf(
                score_text,
                sizeof(score_text),
                "%.2f",
                det.confidence);

            const std::string label =
                class_name + " " + score_text;

            // 绘制检测框
            cv::rectangle(
                annotated,
                rect,
                cv::Scalar(0, 255, 0),
                2);

            int baseline = 0;
            const cv::Size text_size =
                cv::getTextSize(
                    label,
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1,
                    &baseline);

            const int text_y =
                std::max(0, rect.y - text_size.height - 5);

            cv::rectangle(
                annotated,
                cv::Point(rect.x, text_y),
                cv::Point(
                    rect.x + text_size.width,
                    text_y + text_size.height + baseline + 4),
                cv::Scalar(0, 255, 0),
                cv::FILLED);

            cv::putText(
                annotated,
                label,
                cv::Point(rect.x, text_y + text_size.height + 2),
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                cv::Scalar(0, 0, 0),
                1);

            // Detection2D 消息
            vision_msgs::Detection2D detection;
            detection.header = image_msg->header;

            detection.bbox.center.x =
                det.box.x + det.box.width * 0.5;

            detection.bbox.center.y =
                det.box.y + det.box.height * 0.5;

            detection.bbox.center.theta = 0.0;

            detection.bbox.size_x = det.box.width;
            detection.bbox.size_y = det.box.height;

            vision_msgs::ObjectHypothesisWithPose hypothesis;
            hypothesis.id = det.class_id;
            hypothesis.score = det.confidence;

            // 2D 检测没有实际的 3D 位姿，使用默认值
            hypothesis.pose.pose.orientation.w = 1.0;

            detection.results.push_back(hypothesis);
            detection_array.detections.push_back(detection);
        }

        // 发布 Detection2DArray
        detection_pub_.publish(detection_array);

        // 发布标注图像
        cv_bridge::CvImage annotated_msg(
            image_msg->header,
            "bgr8",
            annotated);

        image_pub_.publish(annotated_msg.toImageMsg());
    }

    // ================= Process RSS =================

    double getProcessMemoryMB()
    {
        std::ifstream statm("/proc/self/statm");

        long total_pages = 0;
        long resident_pages = 0;

        if (!(statm >> total_pages >> resident_pages))
            return 0.0;

        const long page_size = sysconf(_SC_PAGESIZE);

        return static_cast<double>(resident_pages) *
               page_size / (1024.0 * 1024.0);
    }

    // ================= Image Callback =================

    void imageCallback(
        const sensor_msgs::ImageConstPtr& image_msg)
    {
        const auto now = std::chrono::steady_clock::now();

        // 限频
        if (max_rate_ > 0.0 && has_last_start_)
        {
            const double elapsed =
                std::chrono::duration<double>(
                    now - last_inference_start_).count();

            if (elapsed < 1.0 / max_rate_)
                return;
        }

        if (has_last_start_)
        {
            interval_sum_ +=
                std::chrono::duration<double>(
                    now - last_inference_start_).count();

            interval_count_++;
        }

        last_inference_start_ = now;
        has_last_start_ = true;

        const auto start = std::chrono::steady_clock::now();

        try
        {
            cv_bridge::CvImageConstPtr cv_ptr =
                cv_bridge::toCvShare(image_msg, "bgr8");

            const cv::Mat& image = cv_ptr->image;

            if (image.empty())
                return;

            ROS_INFO_THROTTLE(
                5.0,
                "YOLO输入图像: %dx%d type=%d step=%zu encoding=%s",
                image.cols,
                image.rows,
                image.type(),
                image.step[0],
                image_msg->encoding.c_str());

            int pad_left = 0;
            int pad_top = 0;
            float scale = 1.0f;

            // 预处理
            preprocess(image, pad_left, pad_top, scale);

            // H2D
            const size_t input_bytes =
                input_buffer_.size() * sizeof(float);

            cudaError_t status = cudaMemcpyAsync(
                device_input_,
                input_buffer_.data(),
                input_bytes,
                cudaMemcpyHostToDevice,
                stream_);

            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));

            // TensorRT 推理
#if NV_TENSORRT_MAJOR >= 10
            if (!context_->enqueueV3(stream_))
                throw std::runtime_error("TensorRT enqueueV3 执行失败");
#else
            std::vector<void*> bindings(engine_->getNbBindings(), nullptr);
            bindings[input_index_] = device_input_;
            bindings[output_index_] = device_output_;

            if (!context_->enqueueV2(bindings.data(), stream_, nullptr))
                throw std::runtime_error("TensorRT enqueueV2 执行失败");
#endif

            // D2H
            const size_t output_bytes =
                output_buffer_.size() * sizeof(float);

            status = cudaMemcpyAsync(
                output_buffer_.data(),
                device_output_,
                output_bytes,
                cudaMemcpyDeviceToHost,
                stream_);

            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));

            if (cudaStreamSynchronize(stream_) != cudaSuccess)
                throw std::runtime_error("CUDA Stream 同步失败");

            // 后处理
            const std::vector<Detection> detections =
                postprocess(image, pad_left, pad_top, scale);

            // 发布两个 ROS 话题
            publishResults(image_msg, image, detections);

            frame_count_++;

            const auto end = std::chrono::steady_clock::now();

            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    end - start).count();

            const double log_window =
                std::chrono::duration<double>(
                    end - last_log_time_).count();

            if (log_window >= 5.0)
            {
                const double fps = frame_count_ / log_window;

                const double avg_interval_ms =
                    interval_count_ > 0
                    ? interval_sum_ / interval_count_ * 1000.0
                    : 0.0;

                ROS_INFO(
                    "YOLO运行中 | 检测数=%zu | 处理耗时=%.1fms "
                    "| FPS=%.2f | 平均识别间隔=%.1fms "
                    "| RSS=%.2fMB",
                    detections.size(),
                    elapsed_ms,
                    fps,
                    avg_interval_ms,
                    getProcessMemoryMB());

                frame_count_ = 0;
                interval_sum_ = 0.0;
                interval_count_ = 0;
                last_log_time_ = end;
            }
        }
        catch (const cv_bridge::Exception& e)
        {
            ROS_ERROR_THROTTLE(
                5.0, "图像转换失败: %s", e.what());
        }
        catch (const std::exception& e)
        {
            ROS_ERROR_THROTTLE(
                5.0, "YOLO推理失败: %s", e.what());
        }
    }
};

// ================= Main =================

int main(int argc, char** argv)
{
    ros::init(argc, argv, "oak_yolo_detection_cpp");

    try
    {
        Yolo26Node node;
        ros::spin();
    }
    catch (const std::exception& e)
    {
        ROS_FATAL("YOLO节点启动失败: %s", e.what());
        return 1;
    }

    return 0;
}
