# Focal Length 실험 설정 정리 (학습/추론)

본 문서는 **focal length 실험**에 한정하여, 논문의 실험 디테일(학습/추론 설정, 파라미터 동결/업데이트 범위)을 코드/설정 파일 기준으로 정리한다.

## 1) 학습(Training) 설정

기준 파일
- `configs/train_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml`
- `train_focal_length.py`

### 1-1. Optimizer / LR / Scheduler
- Optimizer: `AdamW`
- Learning rate: `1e-4` (YAML에서 override)
- Adam betas: `(0.9, 0.999)`
- Adam weight decay: `1e-2`
- Adam epsilon: `1e-8`
- LR scheduler: `constant`
- LR warmup steps: `0`

### 1-2. 배치/학습 스텝 관련
- `train_batch_size: 2`
- `gradient_accumulation_steps: 1`
- `max_train_epoch: 1`
- `max_train_steps: -1` (에폭 길이 기반으로 계산)
- `checkpointing_steps: 1000`
- `mixed_precision_training: true`
- `global_seed: 42`

### 1-3. 데이터/입력 크기
- train/val 모두 `sample_n_frames: 7`
- `sample_size: [256, 384]`
- focal length 학습 데이터셋 클래스: `CameraFocalLength`

### 1-4. focal length 관련 카메라 조건 인코딩
- `camera_encoder_kwargs.downscale_factor: 8`
- `camera_encoder_kwargs.channels: [320, 640, 1280, 1280]`
- `camera_encoder_kwargs.nums_rb: 2`
- `camera_encoder_kwargs.cin: 384` (코드에서 데이터셋 임베딩 채널 및 downscale factor로 재계산되어 사용)
- attention processor 조건:
  - `add_spatial: false`
  - `add_temporal: true`
  - `camera_feature_dimensions: [320, 640, 1280, 1280]`
  - `query_condition: true`
  - `key_value_condition: true`

## 2) 학습 시 동결/업데이트 파라미터

기준 파일: `train_focal_length.py`

### 2-1. 동결(Frozen)
- `vae` 전체 (`requires_grad_(False)`)
- `text_encoder` 전체 (`requires_grad_(False)`)
- `unet` 전체를 우선 동결 (`requires_grad_(False)`)
- `spatial_attn_proc_modules` 내부 파라미터 중 이름에 `'lora'` 포함된 파라미터는 다시 강제 동결

### 2-2. 학습 업데이트(Trainable)
- `camera_encoder` 전체 (`requires_grad_(True)`)
- `unet.attn_processors` / `unet.mm_attn_processors` 중 기본 `AttnProcessor`/`CustomizedAttnProcessor`가 아닌 모듈들을 trainable로 설정
- 실제 optimizer에 전달되는 UNet 파라미터는
  - `requires_grad=True`
  - 파라미터명에 `'merge'` 포함
  - 파라미터명에 `'lora'` 미포함
  조건을 만족하는 항목만 포함

즉, 최종 업데이트 대상은 **(1) camera encoder 파라미터 + (2) 선택된 attention processor의 merge 계열 파라미터(lora 제외)** 이다.

## 3) 추론(Inference) 설정

기준 파일
- `configs/inference_genphoto/adv3_256_384_genphoto_relora_focal_length.yaml`
- `inference_focal_length.py`

### 3-1. 샘플링/추론 하이퍼파라미터
- Scheduler: `DDIMScheduler`
- `num_inference_steps: 25`
- `guidance_scale: 8.0`
- 시드: `torch.manual_seed(42)`
- 기본 생성 해상도/프레임: `height=256`, `width=384`, `video_length=7`

### 3-2. focal length 임베딩 생성
- focal length 시퀀스로부터 FOV 기반 crop ratio를 계산해 binary mask 형태의 focal-length embedding 생성
- 프롬프트 임베딩 차분 기반 CCL embedding과 concat하여 camera embedding 구성

### 3-3. 추론 시 파라미터 동결
- `vae`, `text_encoder`, `unet`, `camera_encoder`, `camera_adaptor` 모두 `requires_grad_(False)`
- 추론 시 optimizer/파라미터 업데이트 없음

## 4) 주의사항 (체크포인트 키 이름)

- 추론 YAML에는 `camera_adpator_ckpt`(오탈자) 키가 기재되어 있다.
- 반면 추론 코드(`inference_focal_length.py`)는 `cfg.camera_adaptor_ckpt`를 참조한다.
- 실행 환경에서 별도 보정이 없다면 focal length adaptor ckpt 로딩이 누락될 수 있으므로 키 정합성 확인이 필요하다.
