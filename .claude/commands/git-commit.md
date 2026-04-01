---
description: Create git commits from unstaged and staged changes
model: claude-sonnet-4-5
---

Given the current repository's unstaged and staged changes, analyze the recent context in this conversation (if any) and create git commits for this work.

Commit messages should be concise yet thorough, up to a maximum of 100 words.
Do not include Claude attribution in commit messages.
IMPORTANT: ensure a clear separation of concerns for the changes in each commit.

Follow these steps for each separate concern/topic: 

1. ONLY stage the changes and untracked files relevant to the topic.
2. Create a git commit with message adhering to the rules mentioned above. 

