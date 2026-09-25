import json
from pathlib import Path
B=Path(__file__).parent
R={r['video_id']:r for r in json.loads((B/'records.json').read_text())}
def save(group, specs):
 out=[]
 for vid,status,note,cl in specs:
  r=R[vid]; claims=[]
  for quote,typ,judgment,interval in cl:
   seg=next((s for s in r['segments'] if quote in s['reason']),None)
   assert seg is not None,(vid,quote)
   assert quote in r['raw_output'],(vid,quote)
   claims.append(dict(quote=quote,claim_type=typ,judgment=judgment,claimed_interval=[seg['start_time'],seg['end_time']],source_interval=interval,observation_vi=note,evidence_images=r['source_sheets']))
  out.append(dict(video_id=vid,status=status,summary_vi=note,claims=claims,inspected_images=r['model_sheets']+r['source_sheets'],model_input_assessment_vi='Đã xem đủ 16 frame model và toàn bộ bảng frame nguồn 2 fps + frame cuối. '+note,reviewer='parent-group-'+str(group)))
 assert len(out)==16
 (B/f'annotations-group-{group}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 save(0,[
 ('54850','confirmed','Khung kim loại có giỏ của xe đẩy hàng, quần xanh và giày trong cú lộn camera; không phải khung xe đạp xanh.', [('green bike frame','wrong_object','contradicted',[0,1.2])]),
 ('52468','ambiguous','Montage diễn viên phát biểu trên sân khấu và nhiều đoạn phim ngắn. Không xác minh được chi tiết góc quay sau lưng và khán giả; không dùng sự vắng mặt ở frame thưa để kết luận bịa.', [('a rear view','unsupported_specificity','ambiguous',[0,26.6])]),
 ('19266','confirmed','Taxi vàng đứng tại chỗ cạnh nhóm người giằng co/ngã và xe máy; không có taxi tăng tốc kéo người phụ nữ khỏi khung hình. Cuối clip chuyển chữ credits.', [('the car’s sudden, violent acceleration','invented_event','contradicted',[4,11.3])]),
 ('52051','confirmed','Cặp đôi trên ván trượt, ngồi rồi đứng ôm nhau ngoài trời. Cảnh cuối là một cô gái cầm cốc đồ ăn; không có cặp đôi áp vào cửa sổ ô tô.', [('the couple is now pressed together against a car window','invented_event','contradicted',[8,11.3])]),
 ('52071','ambiguous','Tay bóp, kéo vật liệu slime tím/cam trong cốc trong suốt. Chất liệu cốc và động tác khuấy đầu đoạn chưa đủ rõ; chuyển động cơ bản có thật.', [('transparent glass mug','unsupported_specificity','ambiguous',[0,22.8])]),
 ('00076','confirmed','Người dùng thanh kiếm đánh chai Diet Coke trên ghế gỗ; lưỡi kiếm cong, chai bị hất lên. Model đổi kiếm thành vòi tưới và đòn đánh thành tia nước.', [('the hand holding the garden hose above the bottle','wrong_object','contradicted',[0,4]),('the water stream suddenly transforms into a powerful, upward jet','invented_event','contradicted',[4,9.5])]),
 ('07039','confirmed','Camera di chuyển vòng quanh tác phẩm dây đen tạo ảo giác hình học; vị trí tranh, cửa và nền thay đổi rõ. Hình học thay đổi theo góc nhìn, không phải camera cố định.', [('The static camera and unchanging background','wrong_attribute','contradicted',[0,12.6])]),
 ('55353','confirmed','Cô gái nhảy trong phòng rồi cắt sang hình mặt sọ toàn màn hình. Chữ nói tiếng gõ không phải cảnh một cánh cửa/lỗ mắt thần.', [('the focus shifts to a door with a peephole','wrong_object','contradicted',[12.5,17.9])]),
 ('25772','ambiguous','Va chạm đường cao tốc, bụi/mảnh vỡ, xe trượt ngang và lật có thật. Cơ chế va đập vào rào, kính và tia lửa ở hình nhỏ chưa đủ rõ. Câu coi đi bằng bốn bánh là phi vật lý là lỗi lập luận, không tự động là bịa hình ảnh.', [('shattering glass','unsupported_specificity','ambiguous',[0,17])]),
 ('00405','confirmed','Người áo đỏ và người áo xám đối diện rồi đánh/đẩy nhau; túi rơi nằm trên sàn. Model kể họ đi sát nhau lãng mạn rồi được bạn an ủi, trái với động tác xung đột kéo dài.', [('walking closely side by side','invented_event','contradicted',[0,5.5]),('her friend rushing up to comfort her','invented_event','contradicted',[7.5,14])]),
 ('21885','confirmed','Cảnh cuối rất ngắn có xe thể thao màu đỏ trên xe cứu hộ, model gọi màu bạc. Cảnh này có cả trong frame cuối model; mốc 7.5s quá sớm là lỗi thời gian riêng.', [('a silver Ferrari','wrong_attribute','contradicted',[10.1,10.4])]),
 ('54226','confirmed','Người áo đen đánh người áo xám ngã, rồi vẫn đứng; người áo tím đứng bên trái làm người chứng kiến. Reasoning gán nạn nhân cho áo tím và mô tả bị ghì trên sàn.', [('the man in the dark purple shirt being forcefully slammed and pinned','wrong_object','contradicted',[3.5,7.9])]),
 ('06773','confirmed','MV có người đàn ông áo trắng, người phụ nữ mặc váy tối, ghế sofa và chùm bóng bay đỏ/trắng sau ghế. Model nhận chùm bóng phía sau thành găng boxing đỏ.', [('red boxing gloves','wrong_object','contradicted',[17.5,23.5])]),
 ('51521','no_clear_error','Người phụ nữ bế em bé rồi chuyển sang trẻ áo hồng ở cầu trượt. Reasoning bỏ qua cú ngã xuống cỏ, nhưng không trực tiếp khẳng định đáp an toàn; không tính bỏ sót hoặc suy diễn cảm xúc là bịa.', []),
 ('33989','ambiguous','Ảnh chó sói nhồi/biến dạng, chữ Nga rồi cảnh thú tối với hiệu ứng kéo méo. Loài vật và tính chất ánh sáng cuối đoạn chưa chắc chắn; không xác nhận bịa chỉ từ hình méo.', [('a bear','unsupported_specificity','ambiguous',[3.8,8.3])]),
 ('05367','confirmed','Nhóm người nhảy trên sân, khoảng 9s mặt sân sập thành hố và nhiều người rơi xuống. Không có voi bước/giẫm người; chuyển động nhìn thấy là người rơi theo nền sập.', [('the large elephant’s forceful trampling of the woman','invented_event','contradicted',[8.5,15.1])]),
 ])
