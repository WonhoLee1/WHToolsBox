# 지난 번에 control center / pyvista를 이용한 pos 도 예상하도록 했다

# 이번에는 구성한 쿠션, 종이 박스, chassis, opencell 구조를 openradioss part 파일로 만들고 메쉬도 생성하고 type25 접촉 조건도 넣고 물성 정보도 생성시켜서 openradioss 해석이 가능한 작업 준비를 하는 기능을 구현해보자

# ohmyclaude의 에이전트로 진행하자

# 작업 폴더는 결과 폴더 구조 하위에 rad 라는 폴더를 만들어 그 내에서 준비한다

# openradioss 동작 여부 판단을 위한 openradioss 설치 폴더는 D:\OpenRadioss_win64\OpenRadioss이며, 실행 파일들은 D:\OpenRadioss_win64\OpenRadioss\exec 폴더 아래에 위치

# 실행과 관련된 py 파일들은 D:\OpenRadioss_win64\OpenRadioss\openradioss_gui

# skills에 whtools-openradiossent 폴더의 내용을 참고하여, rag을 충실히 활용하세요

# wht_openradiossent_doc에 각종 필요파일들 위치

# 원점에 파트들의 중심 위치를 잡았을 때, 옵션에 따라서, 지면이 회전 후 이동되는 구조이거나, 파트들이 회전-이동 되는 구조로 구성할 수 있다

# 지면은 두께 10 mm 의 육면체로 구성하자

# 각 파트들의 회전, 병진 변화를 시키는 방법은 원점에서 생성된 파트들을 openradioss (radioss)의 /TANSFORM 키워드를 사용해서 초기 자세를 세팅하자

# 접촉은 type 25로 set으로 전체 파트를 설정해서... slave로만 지정하면 됨

# 중력 가속도 -9806 mm/s

# input radioss cards (start, engine)을 생성

# openradioss용 모델 생성에는 gmsh를 적극적으로 활용

# box 도 생성한다.  비록, 모션 시뮬레이션 모델에서는 사용하지 않더라도, radioss model은 생성한다

# part별로 inc 확장자 파일을 만들고 main radioss 파일에서 #include 하는 구조도 좋다

# openradioss 모델 생성의 호출은 control center의 Log Motion 버튼 콤보항목에 Create a OpenRadioss Model과 Run a OpenRadioss Model로 한다

# 더 확인이 필요한 내용이 있으면 인터뷰해달라

# radioss export 시에 세트를 회전병진이동할지, 바닥을 회전병진운동할지를 결정하는 flag가 있어야 할 것 같아

# 그 기준에 맞춰서 세트 전체에 병진속도, 회전각속도의 초기 상태를 부여하는 radioss 키워드도 포함시켜라

# 모델의 input 파일을 내용을 보는 것이 불편하다. 왜냐면 적절한 viewer가 시중에 없다

# 따라서, ls-dyna 양식으로도 k 파일을 출력하자. radioss 카드와 1:1로 맞는 카드를 사용할 수 있다
