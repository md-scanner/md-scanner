#include <opencv2/opencv.hpp>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

using namespace cv;
using namespace std;

enum bb_type_t {
    none = 0,
    paragraph = 1,
    h1 = 2,
    code = 3,
    ul = 4,
    img = 5,
    ol = 6,
    h2 = 7,
    h3 = 8,
    h4 = 9,
    h5 = 10,
    h6 = 11
};

// colors for BBs of various kinds of sections, as generated by md-rendered
const map<bb_type_t, Vec3b> colors = {
        {paragraph, {255, 0, 0}},
        {code, {0, 255, 0}},
        {h1, {0, 0, 255}},
        {h2, {255, 0, 255}},
        {h3, {255, 0x10, 0xAA}},
        {h4, {255, 0xBB, 0x10}},
        {h5, {255, 255, 0}},
        {h6, {255, 0x90, 0}},
        {img, {0x50, 0xAA, 0x90}},
        {ul, {0, 255, 255}},
        {ol, {0xCB, 0xC0, 255}}
};

// maximum distance acceptable to consider a pixel of the color of a BB
const double DIST_THRESH = 8;


double l2_distance(Vec3b a, Vec3b b) {
    return sqrt(pow(a[0]-b[0], 2) + pow(a[1]-b[1], 2) + pow(a[2]-b[2], 2));
}

// given a pixel, compute whether it could or could not be part of a bounding box
bb_type_t get_pixel_bb_type(const Vec3b &p) {
    for(const auto &type : colors) {
        if(l2_distance(p, type.second) < DIST_THRESH) {
            return type.first;
        }
    }
    return none;
}

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

    // find start y of bounding box
    for(ret.start_y = start; ret.start_y < img.rows; ret.start_y++) {
        Vec3b p = img.at<Vec3b>(ret.start_y, col);

        ret.type = get_pixel_bb_type(p);

        if(ret.type != none) break;
    }

    if(ret.type == none) return ret;

    // find start x
    
    /* comparison to col is because ret.start_x is unsigned, so we need to check
     * for overflow
     */
    for(ret.start_x = col; ret.start_x <= col; ret.start_x--) {
        Vec3b p = img.at<Vec3b>(ret.start_y, ret.start_x);
        bb_type_t type = get_pixel_bb_type(p);

        if(ret.type != type) {
            ret.start_x++; // going back since the loop stops after the end
            break;
        }
    }

    // find end x
    for(ret.end_x = col; ret.end_x < img.cols; ret.end_x++) {
        Vec3b p = img.at<Vec3b>(ret.start_y, ret.end_x);
        bb_type_t type = get_pixel_bb_type(p);

        if(ret.type != type) {
            ret.end_x--; // going back since the loop stops after the end
            break;
        }
    }

    // find end y: go through both the leading and trailing edge
    // look at the leading edge
    for(ret.end_y = ret.start_y; ret.end_y < img.rows; ret.end_y++) {
        Vec3b p = img.at<Vec3b>(ret.end_y, ret.start_x);
        bb_type_t type = get_pixel_bb_type(p);

        if(ret.type != type) {
            size_t start_x = ret.start_x;
            bool found = false;
            while(start_x > 0) {
                start_x--;
                p = img.at<Vec3b>(ret.end_y, start_x);
                if(get_pixel_bb_type(p) == ret.type) {
                    ret.start_x = start_x;
                    found = true;
                    break;
                }
            }
            if(!found) {
                ret.end_y--; // same reason as previously
                break;
            }
            
        }
    }

    // deduplicate bbs: scroll down until there the bb ends

    while(
        ret.end_y < img.rows &&
        get_pixel_bb_type(img.at<Vec3b>(ret.end_y, col)) == ret.type
    ) {
        ret.end_y++;
    }


    /* look at the trailing edge
    for(; ret.end_y < img.rows; ret.end_y++) {
        Vec3b p = img.at<Vec3b>(ret.end_y, ret.end_x);
        bb_type_t type = get_pixel_bb_type(p);

        if(ret.type != type) {
            printf("END: Risking crashing out at y=%d\n", ret.end_y);
            size_t end_x = ret.start_x;
            bool found = false;
            while(end_x < img.cols) {
                end_x++;
                p = img.at<Vec3b>(ret.end_y, end_x);
                if(get_pixel_bb_type(p) == type) {
                    ret.end_x = end_x;
                    found = true;
                    break;
                }
            }
            if(!found) {
                ret.end_y--; // same reason as previously
                printf("END: Crashing out at y=%d\n", ret.end_y);
                break;
            }
        }
    }*/




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