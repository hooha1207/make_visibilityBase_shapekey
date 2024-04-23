# make_visibilityBase_shapekey




일반적으로 bone을 이용해서 변형할 수 있는 mesh의 반경은 한정되어 있다
(추가 bone 설치 후 특정 조건을 만들지 않는다는 전제조건 하에)

때문에 bone으로 mesh를 변형시킨 다음 modifier의 on cage, edit mode 설정을 활성화하여
modifier가 적용된 evaluated_depsgraphs mesh에서 mesh를 변형할 수 있게 바꾼다

이후 shapekey를 추가해서 부족한 형태를 보충하도록 shapekey를 만들어
bone에 driver로 연결해 bone deform으로는 만들 수 없던 형태를 만들 게 된다


해당 프로세스의 단점
:
weight paint가 바뀌면 shapekey도 바뀌므로 이에 맞게 수정해줘야 됨
weight가 바뀌면 shapekey도 바뛰어야 되기 떄문에 weight를 가변적으로 정의할 수 없다
절대 형태를 타겟으로 shapekey를 만들기 때문에 정확하지 않으며, 심지어 snap 기능도 쓸 수 없어 사용자가 눈대중으로 맞춰야 된다.


위 단점은 vertex를 전부 target mesh에 맞게 사용자가 직접 옮겨야 되기 때문에
작업시간이 굉장히 길어지는 문제가 존재한다

이를 해결하기 위해
절대형태 mesh를 기준으로 bone deform으로 변형된 mesh가 어떤 모양의 shapekey를 가져야만 되는지를
딥러닝과 같이 편미분으로 계산해서 완벽하지는 않지만 근사값을 구하는 스크립트를 만들어야한다

해당 repository는 이 스크립트를 구현하기 위해 작성되었다.