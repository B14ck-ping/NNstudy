#include "processor.hpp"




static void s_proc_thread();

processor *processor::Instance()
{

    static processor* _instance = 0;
    if (_instance == 0){
        _instance = new processor;
        const unsigned processor_count = std::thread::hardware_concurrency();
        _instance->is_active = true;
        cout << "Found " << processor_count << " CPUs" << endl;

        for (unsigned i = 0; i < processor_count; i++){
            thread t(s_proc_thread);
            t.detach();
        }
    }

    return _instance;
}

processor::processor()
{
    // const unsigned processor_count = std::thread::hardware_concurrency();
    // is_active = true;
    // cout << "Found " << processor_count << " CPUs" << endl;

    // for (unsigned i = 0; i < processor_count; i++){
    //     thread t(s_proc_thread);
    //     t.detach();
    // }
}


static void s_proc_thread()
{
    processor* const p = processor::Instance();

    while(p->isActive()){
        p->procQueueLock();
        if (!p->proc_queue.empty()){
            std::function<void()> func = p->proc_queue.front();
            p->proc_queue.pop();
            func();
        } else {
            p->procQueueUnlock();
            continue;
        }
        p->procQueueUnlock();


    }

    cout << "Thread is finished!" << endl;
}

template<typename _Callable, typename... _Args>
void processor::QueueFunction(_Callable&& __f, _Args&&... __args)
{
    processor* const p = processor::Instance();
    std::function<void()> func = std::bind(std::forward<_Callable>(__f), std::forward<_Args>(__args)...);
    p->procQueueLock();
    proc_queue.push(func);
    p->procQueueUnlock();
}