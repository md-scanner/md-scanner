#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include <sstream>

using namespace cv;
using namespace std;

/* Functions to compare pixel values to bounding-box colors.
 * Comparison is approximate to avoid issues with compression.
 * OpenCV uses BGR
 */

inline bool is_blue(Vec3b p) {
    return p[0] >= 0xFD && p[1] < 0x10 && p[2] < 0x10;
}

inline bool is_red(Vec3b p) {
    return p[0] < 0x10 && p[1] < 0x10 && p[2] >= 0xFD;
}

inline bool is_green(Vec3b p) {
    return p[0] < 0x10 && p[1] >= 0xFD && p[2] < 0x10;
}


enum bb_type_t {
    paragraph, title, code, none
};

struct bb_result_t {
    bb_type_t type = none;
    size_t start_x = 0;
    size_t start_y = 0;
    size_t end_x = 0;
    size_t end_y = 0;
};

/* Function that finds the next bounding box for a title, paragraph or code
 * section.
 * 
 * The function scans along a vertical line at column col, and looks for matches
 * in the color to the one that the BBs should have. If it finds one, it looks
 * for the boundaries of the BB and returns them, along with their type.
 * 
 * Keep col at 1 << 31 to use the middle of the image as the starting point.
 * 
 * 2^31 was chosen because it's easy to compute, way too large to be realistic,
 * and representable as a size_t with some margin in 32-bit environments as well
 */
bb_result_t find_next_bb(Mat &img, size_t start = 0, size_t col = 1 << 31) {
    bb_result_t ret;
    // we assume 2^31 is not a realistic value
    // and use the middle of the image instead
    if(col == 1 << 31) {
        col = img.cols / 2;
    }

    for(ret.start_y = start; ret.start_y < img.rows; ret.start_y++) {
        /* the colors used for the bounding boxes are:
         * * blue for paragraphs
         * * red for titles
         * * green for code (?)
         */

        Vec3b p = img.at<Vec3b>(ret.start_y, col);

        if(is_blue((p))) {
            ret.type = paragraph;
            break;
        } else if(is_green(p)) {
            ret.type = code;
            break;
        } else if(is_red(p)) {
            ret.type = title;
            break;
        }
    }

    if(ret.type == none) return ret;

    // find start x
    
    /* comparison to col is because ret.start_x is unsigned, so we need to check
     * for overflow
     */
    for(ret.start_x = col; ret.start_x <= col; ret.start_x--) {
        Vec3b p = img.at<Vec3b>(ret.start_y, ret.start_x);
        if(
            (ret.type == paragraph && !is_blue(p)) ||
            (ret.type == code && !is_green(p)) ||
            (ret.type == title && !is_red(p))
        ) {
            ret.start_x++; // going back since the loop stops after the end
            break;
        }
    }

    // find end x
    for(ret.end_x = col; ret.end_x < img.cols; ret.end_x++) {
        Vec3b p = img.at<Vec3b>(ret.start_y, ret.end_x);
        if(
            (ret.type == paragraph && !is_blue(p)) ||
            (ret.type == code && !is_green(p)) ||
            (ret.type == title && !is_red(p))
        ) {
            ret.end_x--; // going back since the loop stops after the end
            break;
        }
    }

    // find end y
    for(ret.end_y = ret.start_y; ret.end_y < img.rows; ret.end_y++) {
        Vec3b p = img.at<Vec3b>(ret.end_y, ret.start_x);
        if(
            (ret.type == paragraph && !is_blue(p)) ||
            (ret.type == code && !is_green(p)) ||
            (ret.type == title && !is_red(p))
        ) {
            ret.end_y--; // same reason as above
            break;
        }
    }



    return ret;
}

int main(int argc, char **argv) {
    if(argc != 2) {
        puts("Usage:\n\t./extract-bbs IMAGE_PATH");
        return 1;
    }

    Mat img = imread(argv[1], IMREAD_COLOR);

    if(img.empty()) {
        puts("Unable to read image");
        return 2;
    }

    stringstream outfile_ss(argv[1], ios_base::app | ios_base::out);

    outfile_ss << ".csv";

    ofstream os(outfile_ss.str());

    // writing the CSV header
    os << "Type,Start X,Start Y,End X,End Y\n";

    bb_result_t res;
    do {
        res = find_next_bb(img, res.end_y+1);
        
        if(res.type != none) {
            // writing the CSV manually (easier and more compact than JSON)
            os << res.type << "," << res.start_x << "," << res.start_y << ","
               << res.end_x << "," << res.end_y << "\n";
        }


    } while(res.type != none);

}