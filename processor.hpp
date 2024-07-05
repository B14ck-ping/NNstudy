#pragma once
#include <iostream>
#include <queue>
#include <thread>
#include <mutex> 
#include <functional>
#include <future>

using namespace std;



class processor {
private:
    explicit processor();
    processor(processor&) = delete;
    processor(const processor&) = delete;
    ~processor(){};
    mutex proc_queue_mutex;
    mutex processor_mutex;

public:
    bool is_active;
    std::queue<std::function<void()>> proc_queue;
    static processor* Instance();
    bool isActive() const {return is_active;}
    void procQueueLock() {proc_queue_mutex.lock();}
    void procQueueUnlock() {proc_queue_mutex.unlock();}
    
    template<typename _Callable, typename... _Args>
    void QueueFunction(_Callable&& __f, _Args&&... __args);

};