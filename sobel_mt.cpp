#include <stdio.h>
#include <stdlib.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <fstream>
#include <errno.h>

#include <unistd.h>
#include <string.h>
#include <locale.h>
#include <sys/ioctl.h>
#include <err.h>

#include "sobel_alg.h"
#include "pc.h"

// Replaces img.step[0] and img.step[1] calls in sobel calc

using namespace cv;

static ofstream results_file;

// Define image mats to pass between function calls
static Mat img_gray, img_sobel, src;
bool quit;
static float total_fps, total_ipc, total_epf;
static float gray_total, sobel_total, cap_total, disp_total;
static float sobel_ic_total, sobel_l1cm_total;

pthread_barrier_t barrScale;	// barrier for grayScale
pthread_barrier_t barrCalc;		// barrier for sobelCalc
pthread_barrier_t barrShow;		// barrier for imshow
pthread_barrier_t barrLoop;		// barrier for loop end

/*******************************************
 * Model: runSobelMT
 * Input: None
 * Output: None
 * Desc: This method pulls in an image from the webcam, feeds it into the
 *   sobelCalc module, and displays the returned Sobel filtered image. This
 *   function processes NUM_ITER frames.
 ********************************************/
void *runSobelMT(void *ptr) {
	pthread_t myID = pthread_self();
	// Allow the threads to contest for thread0 (controller thread) status
	pthread_mutex_lock(&thread0);

	// Check to see if this thread is first to this part of the code
	if (thread0_id == 0) {
		pthread_barrier_init(&barrScale, NULL, 2);
		pthread_barrier_init(&barrCalc, NULL, 2);
		pthread_barrier_init(&barrShow, NULL, 2);
		pthread_barrier_init(&barrLoop, NULL, 2);
		quit = false;
		thread0_id = myID;
	}
	pthread_mutex_unlock(&thread0);

	// second thread only performs half of grayScale and sobelCalc
	if (myID != thread0_id) {
		while(quit == false) {
			pthread_barrier_wait(&barrScale);
			grayScale(src, img_gray, 2);
			pthread_barrier_wait(&barrCalc);
			int i = myID;
			int j = 0;
			for(i = 7;i <100; i++) {
				if(i % 13 == 3) j+= 3;
				j++;
			}
			sobelCalc(img_gray, img_sobel, 2);
			pthread_barrier_wait(&barrShow);
			pthread_barrier_wait(&barrLoop);
		}
		pthread_barrier_wait(&endSobel);
		return NULL;
	}

	// Set up variables for computing Sobel
	string top = "Sobel Top";
	uint64_t cap_time, gray_time, sobel_time, disp_time, sobel_l1cm, sobel_ic;
	counters_t perf_counters;


	pc_init(&perf_counters, 0);

	// Start algorithm
	CvCapture* video_cap;

	if (opts.webcam) {
		video_cap = cvCreateCameraCapture(-1);
	} else {
		video_cap = cvCreateFileCapture(opts.videoFile);
	}

	video_cap = cvCreateFileCapture(opts.videoFile);
	cvSetCaptureProperty(video_cap, CV_CAP_PROP_FRAME_WIDTH, IMG_WIDTH);
	cvSetCaptureProperty(video_cap, CV_CAP_PROP_FRAME_HEIGHT, IMG_HEIGHT);

	// Keep track of the frames
	int i = 0;
	img_gray = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);
	img_sobel = Mat(IMG_HEIGHT, IMG_WIDTH, CV_8UC1);

	while (1) {
		// Allocate memory to hold grayscale and sobel images

		pc_start(&perf_counters);
		src = cvQueryFrame(video_cap);
		pc_stop(&perf_counters);

		cap_time = perf_counters.cycles.count;
		sobel_l1cm = perf_counters.l1_misses.count;
		sobel_ic = perf_counters.ic.count;

		// LAB 2, PART 2: Start parallel section
		// barrier here for guaranteed src
		pthread_barrier_wait(&barrScale);
	 	pc_start(&perf_counters);
		grayScale(src, img_gray, 1);
		pc_stop(&perf_counters);

		gray_time = perf_counters.cycles.count;
		sobel_l1cm += perf_counters.l1_misses.count;
		sobel_ic += perf_counters.ic.count;

		// barrier here, so threads are guaranteed img_gray data
		pthread_barrier_wait(&barrCalc);
		pc_start(&perf_counters);
		sobelCalc(img_gray, img_sobel, 1);
		pc_stop(&perf_counters);

		sobel_time = perf_counters.cycles.count;
		sobel_l1cm += perf_counters.l1_misses.count;
		sobel_ic += perf_counters.ic.count;
		// LAB 2, PART 2: End parallel section

		// barrier here for guaranteed img_sobel
		pthread_barrier_wait(&barrShow);
		pc_start(&perf_counters);
		namedWindow(top, CV_WINDOW_AUTOSIZE);
		imshow(top, img_sobel);
		pc_stop(&perf_counters);

		disp_time = perf_counters.cycles.count;
		sobel_l1cm += perf_counters.l1_misses.count;
		sobel_ic += perf_counters.ic.count;

		cap_total += cap_time;
		gray_total += gray_time;
		sobel_total += sobel_time;
		sobel_l1cm_total += sobel_l1cm;
		sobel_ic_total += sobel_ic;
		disp_total += disp_time;
		total_fps += PROC_FREQ/float(cap_time + disp_time + gray_time + sobel_time);
		total_ipc += float(sobel_ic/float(cap_time + disp_time + gray_time + sobel_time));
		i++;

		// Press q to exit
		char c = cvWaitKey(10);
		if (c == 'q' || i >= opts.numFrames) {
			quit = true;
			// sync even when exiting
			pthread_barrier_wait(&barrLoop);
  			break;
		}
		// barrier here for syncing quit with other thread
		pthread_barrier_wait(&barrLoop);
	}

  total_epf = PROC_EPC*NCORES/(total_fps/i);
  float total_time = float(gray_total + sobel_total + cap_total + disp_total);

  results_file.open("mt_perf.csv", ios::out);
  results_file << "Percent of time per function" << endl;
  results_file << "Capture, " << (cap_total/total_time)*100 << "%" << endl;
  results_file << "Grayscale, " << (gray_total/total_time)*100 << "%" << endl;
  results_file << "Sobel, " << (sobel_total/total_time)*100 << "%" << endl;
  results_file << "Display, " << (disp_total/total_time)*100 << "%" << endl;
  results_file << "\nSummary" << endl;
  results_file << "Frames per second, " << total_fps/i << endl;
  results_file << "Cycles per frame, " << total_time/i << endl;
  results_file << "Energy per frames (mJ), " << total_epf*1000 << endl;
  results_file << "Total frames, " << i << endl;
  results_file << "\nHardware Stats (Cap + Gray + Sobel + Display)" << endl;
  results_file << "Instructions per cycle, " << total_ipc/i << endl;
  results_file << "L1 misses per frame, " << sobel_l1cm_total/i << endl;
  results_file << "L1 misses per instruction, " << sobel_l1cm_total/sobel_ic_total << endl;
  results_file << "Instruction count per frame, " << sobel_ic_total/i << endl;

  cvReleaseCapture(&video_cap);
  results_file.close();
  pthread_barrier_wait(&endSobel);
  return NULL;
}
