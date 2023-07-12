#include <opencv2/opencv.hpp>
#include <cstdio>

using namespace cv;

enum bb_type_t {
    paragraph, title, code
};

struct bb_result_t {
    bb_type_t type;
    size_t start_x;
    size_t start_y;
    size_t end_x;
    size_t end_y;
};

/* Function that finds the next bounding box for a title, paragraph or code
 * section.
 * 
 * The function scans along a vertical line at column col, and looks for matches
 * in the color to the one that the BBs should have. If it finds one, it looks
 * for the boundaries of the BB and returns them, along with their type.
 * 
 * Keep col at 1 << 31 to use the middle of the image as the starting point.
 */
bb_result_t find_next_bb(Mat img, size_t col = 1 << 31, size_t start = 0) {
    bb_result_t ret;
    // we assume 2^31 is not a realistic value
    // and use the middle of the image instead
    if(col == 1 << 31) {
        col = img.size[1] / 2;
    }

    for(size_t row = start; row < img.size[0]; row++) {
        /* we look at the color approximately in order to avoid issues with
         * rendering and/or compression.
         */

        /* the colors used for the bounding boxes are:
         * * blue for paragraphs
         * * red for titles
         * * green for code (?)
         */

        Vec3b p = img.at<Vec3b>(row, col);

        // OpenCV uses BGR
        if(p[0] >= 0xFD && p[1] < 0x10 && p[2] < 0x10) {
            ret.start_y = row;
            ret.type = paragraph;
        } else if(p[0] < 0x10 && p[1] >= 0xFD && p[2] < 0x10) {
            ret.start_y = row;
            ret.type = code;
        } else if(p[0] < 0x10 && p[1] < 0x10 && p[2] >= 0xFD) {
            ret.start_y = row;
            ret.type = title;
        }
    }

    // find start x

    // find end x and y

    // return value
}

int main(int argc, char **argv) {
    using std::cout;
    if(argc != 2) {
        puts("Usage:\n\t./extract-bbs IMAGE_PATH");
        return 1;
    }

    Mat img = imread(argv[1], IMREAD_COLOR);

    if(img.empty()) {
        puts("Unable to read image");
        return 2;
    }


}