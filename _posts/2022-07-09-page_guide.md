---
layout: single
title: "How to Write Github Page"
---

# gitub page 블로그 작성법


#### 레포 클로닝
- `$ git clone https://github.com/ComTalk/ComTalk.github.io.git`

#### `.md` 로 블로그에 게시할 글 작성
- `yyyy-mm-dd-whatever.md` 형태의 제목으로 작성 `whatever` 에는 파일의 내용을 나타내는 이름으로 작성. url에 쓰임

#### `.md` 상단에 해당 내용 추가
```
---
layout: post 혹은 single
title: 블로그 글 제목
---
```

#### `_posts` 에 `.md` 파일 옮기기
- `$ git commit -m "블로그 글 내용 한줄 요약"`
- `$ git push origin master`
