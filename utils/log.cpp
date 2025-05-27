static void s_print_duration(chrono::microseconds duration)
{
    long long hours, minutes, seconds;

    if (duration.count() < 1000){
        cout << duration.count() << " microseconds" << endl;
        return;
    } else if (duration.count() > 1000 && duration.count() < 1000000){
        cout << duration.count()/1000 << " milliseconds" << endl;
        return;
    } else if (duration.count() > 1000000 && duration.count() < 60000000){
        cout << duration.count()/1000000 << " seconds" << endl;
        return;
    } else if (duration.count() > 60000000 && duration.count() < 3600000000){
        seconds = duration.count()/1000000LL;
        minutes = seconds/60;
        seconds = seconds%60;
        cout << minutes << " minutes " << seconds << " seconds" << endl;
        return;
    } else if (duration.count() > 3600000000){
        seconds = duration.count()/1000000LL;
        minutes = seconds/60;
        seconds = seconds%60;
        hours = minutes/60;
        minutes = minutes%60;
        cout << hours << " hours " << minutes << " minutes " << seconds << " seconds" << endl;
        return;
    }
   
}

void printProgressBar(size_t itemCnt, size_t curr_item)
{
    static int last_percent = 0;
    int curr_percent = (int)(((float)(curr_item+1)/(float)itemCnt)*100.0);
    if (curr_percent == last_percent) return;
    last_percent = curr_percent;
    // cout << "\x1b[2K" << "\x1b[1A" << "\x1b[2K" << "\r";
    cout << "\r";
    cout << "Progress: " << curr_percent << "% ";
    cout << "[";
    for (int j = 0; j < curr_percent; j++){
        cout << "=";
    }

    for (int j = 0; j < 100 - curr_percent; j++){
        cout << "-";
    }
    
    cout << "]" << curr_item+1 << "/" << itemCnt;// << endl;
}

void printMatrix(long r, long c, float *arr)
{
    for (long i = 0; i < r; i++){
        for (long j = 0; j < c; j++){
            std::cout << arr[i*c+j] << ' ';
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
