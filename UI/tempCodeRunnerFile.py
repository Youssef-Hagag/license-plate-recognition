for i in range(len(cur_progress_thread)):
          cur_progress_thread[i].join()
        cur_progress_thread.clear()
        for i in range(len(cur_img_thread)):
          cur_img_thread[i].join()
        cur_img_thread.clear()