# Lessons

- **bkit `missing=[skill_post]` 세션 경고는 거짓 경보다 (CC 한계, bkit 오진).**
  CC는 스킬 호출에 PostToolUse를 발화하지 않는다(스킬=프롬프트 확장, 디스패치 도구 아님 — anthropics/claude-code #43630, "not planned"). 따라서 bkit의 `matcher:"Skill"` 훅(scripts/skill-post.js)은 구조적으로 절대 실행되지 않고, `skill_post` 스탬프가 안 찍혀 매 세션 경고가 뜬다. bkit은 이를 "#57317 일시 드롭 의심"으로 라벨링하지만 실은 영구적. bash_post/write_post(실제 도구 PostToolUse)는 정상 발화하므로 "전체 드롭"이 아님.
  *왜 중요:* 디버깅 시 타임존(audit는 UTC `Z`, `ls`는 +0900 KST)을 혼동하면 "훅이 다 죽었다"고 오판하기 쉽다. 결정적 증거는 audit의 `skill_executed` 카운트=0(skill-post.js가 단 한 번도 실행된 적 없음)였다.
  *해결:* `.claude/hooks/bkit-skill-reachability-shim.js` + `.claude/settings.local.json`의 Stop 훅으로 매 턴 skill_post 스탬프를 갱신해 거짓 경보만 차단(기능 복구 아님 — 이 프로젝트는 PDCA 미사용이라 무영향). bkit이 모니터를 고치거나 CC가 동작을 바꾸면 제거.
