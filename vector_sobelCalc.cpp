		for (int j=1; j<img_gray.cols-1; j+=8) {
			unsigned char *current = &img_data[IMG_WIDTH*i + j];
			// quantity added in both convolutions
			uint8x8_t temp1 = vld1_u8(current - IMG_WIDTH - 1);
			uint8x8_t temp2 = vld1_u8(current + IMG_WIDTH + 1);
			int16x8_t sum = (int16x8_t) vsubl_u8(temp1, temp2);
			// quantity added in x conv, subtracted in y conv
			temp1 = vld1_u8(current - IMG_WIDTH + 1);
			temp2 = vld1_u8(current + IMG_WIDTH - 1);
			int16x8_t diff = (int16x8_t) vsubl_u8(temp1, temp2);
			// temp3 = 2*(arr[curr -IMG] - arr[curr + IMG])
			temp1 = vld1_u8(current - IMG_WIDTH);
			temp2 = vld1_u8(current + IMG_WIDTH);
			int16x8_t temp3 = (int16x8_t) vsubl_u8(temp1, temp2);
			temp3 = vshlq_n_s16(temp3, 1);
		}
