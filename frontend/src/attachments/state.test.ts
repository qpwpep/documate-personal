import { describe, expect, it } from 'vitest';
import { attachmentName } from './state';

describe('서버와 동일한 첨부 이름 비교', () => {
  it.each([
    ['straße.py', 'STRASSE.py'], ['ẞ.py', 'ss.py'], ['ΟΣ.py', 'οσ.py'], ['ΟΣ.py', 'ος.py'],
    ['ﬀ.py', 'ff.py'], ['가.py', '\u1100\u1161.py'],
  ])('%s와 %s는 같은 이름이다', (first, second) => {
    expect(attachmentName(first)).toBe(attachmentName(second));
  });

  it('casefold가 구분하는 dotless i와 ASCII i를 합치지 않는다', () => {
    expect(attachmentName('ı.py')).not.toBe(attachmentName('I.py'));
  });
});
