from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest
api=KaggleApi(); api.authenticate()
r=ApiListKernelSessionOutputRequest(); r.user_name='vucongaaa'; r.kernel_slug='dar-grpo'; r.page_size=200
with api.build_kaggle_client() as client:
    for page in range(5):
        out=client.kernels.kernels_api_client.list_kernel_session_output(r)
        names=[x.file_name for x in out.files or []]
        print('page',page,'count',len(names),'first',names[:1],'last',names[-1:],'token',str(out.next_page_token)[:20],flush=True)
        for item in out.files or []:
            if 'dar_grpo/checkpoints' in item.file_name:
                print(item.file_name, getattr(item,'size',None), getattr(item,'total_bytes',None))
        if not out.next_page_token:
            break
        r.page_token=out.next_page_token
